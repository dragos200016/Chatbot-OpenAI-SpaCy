from .models import ChatMessage
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from .forms import LoginForm, RegistrationForm
from django.http import JsonResponse
from llama_index import ServiceContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms import OpenAI
from . import config
import openai
import time
import spacy
import psutil
import random
from fuzzywuzzy import fuzz

# Initialize SpaCy with English language model
nlp = spacy.load('en_core_web_sm')

openai.api_key = config.OPENAI_API_KEY

# Knowledge base for SpaCy processing
knowledge_base = {
    "author": "J. K. Rowling",
    "characters": ["Harry Potter", "Hermione Granger", "Ron Weasley"],
    "school": "Hogwarts School of Witchcraft and Wizardry",
    "antagonist": "Lord Voldemort",
    "publishing": {
        "UK": "Bloomsbury",
        "US": "Scholastic Press"
    },
    "genres": ["fantasy", "drama", "coming-of-age fiction", "British school story"],
    "themes": ["prejudice", "corruption", "madness", "death"]
}

def display_history(user):
    # Retrieve chat history for the user, including bot messages (where user is None)
    history = ChatMessage.objects.filter(user=user) | ChatMessage.objects.filter(user=None)
    return history

def process_with_spacy(user_input):
    doc = nlp(user_input)

    # Initialize variables to store the subject, predicate, and object
    subject = None
    predicate = None
    obj = None

    # Extract syntactic dependencies to identify subject, predicate, and object
    for token in doc:
        if token.dep_ == "nsubj":
            subject = token.text
        elif token.dep_ == "ROOT":
            predicate = token.text
        elif token.dep_ == "dobj":
            # Handle cases where the object consists of multiple tokens
            obj = " ".join([t.text for t in token.subtree])

    # Check if the user input is a question about the author of a book
    if doc[0].text.lower() in ["who", "what"] and predicate.lower() in ["write", "wrote"] and obj:
        book = obj.lower()
        if "harry potter" in book:
            author = knowledge_base.get("author", "Unknown")
            response = f"The author of {book.title()} is {author}."
        else:
            response = "I'm sorry, I don't know the author of that book."

    # Additional checks for other types of questions
    elif doc[0].text.lower() in ["who", "what"] and predicate.lower() in ["are", "is"]:
        if "main characters" in user_input.lower():
            characters = ", ".join(knowledge_base["characters"])
            response = f"The main characters in Harry Potter are {characters}."
        elif "school" in user_input.lower():
            school = knowledge_base.get("school", "Unknown")
            response = f"The school in Harry Potter is {school}."
        elif "antagonist" in user_input.lower():
            antagonist = knowledge_base.get("antagonist", "Unknown")
            response = f"The main antagonist in Harry Potter is {antagonist}."
        elif "publisher" in user_input.lower():
            publisher_uk = knowledge_base["publishing"].get("UK", "Unknown")
            publisher_us = knowledge_base["publishing"].get("US", "Unknown")
            response = f"In the UK, Harry Potter was published by {publisher_uk}. In the US, it was published by {publisher_us}."
        elif "genres" in user_input.lower():
            genres = ", ".join(knowledge_base["genres"])
            response = f"Harry Potter belongs to the following genres: {genres}."
        elif "themes" in user_input.lower():
            themes = ", ".join(knowledge_base["themes"])
            response = f"The major themes in Harry Potter include: {themes}."
        else:
            response = "I'm sorry, I don't have information on that topic."

    else:
        # If it's not a known question format, return a default response
        response = "I'm sorry, I didn't understand the question."

    return response

def calculate_accuracy(response, expected_response):
    similarity_ratio = fuzz.token_sort_ratio(response.lower(), expected_response.lower())
    return similarity_ratio / 100.0  # Normalize to a 0-1 scale

def measure_resource_consumption():
    # Measure current resource consumption (CPU and memory usage)
    process = psutil.Process()
    cpu_usage = process.cpu_percent(interval=1)
    memory_info = process.memory_info()
    memory_usage = memory_info.rss  # Resident Set Size
    return cpu_usage, memory_usage

def generate_confidence_score():
    # Generate a mock confidence score (as OpenAI GPT-3.5-turbo-instruct does not return confidence scores)
    return random.uniform(0.5, 1.0)  # Mock confidence score between 0.5 and 1.0

@login_required
def delete_chat_history(request):
    if request.method == 'POST':
        # Delete chat messages associated with the current user
        ChatMessage.objects.filter(user=request.user).delete()
        return JsonResponse({'success': True})
    else:
        return JsonResponse({'success': False})

@login_required(login_url='login')
def chat(request):
    if request.method == 'POST':
        query = request.POST.get('query', '')
        current_user = request.user

        # Save the user's message to the database
        user_message = ChatMessage.objects.create(user_id=current_user.id, message=query)

        # Measure start time for SpaCy processing
        start_time_spacy = time.time()
        
        # Process user input with SpaCy
        spacy_response = process_with_spacy(query)
        
        # Measure end time for SpaCy processing
        end_time_spacy = time.time()
        spacy_response_time = end_time_spacy - start_time_spacy  # Calculate time taken for SpaCy
        
        # Measure resource consumption for SpaCy
        spacy_cpu_usage, spacy_memory_usage = measure_resource_consumption()
        
        # Generate mock confidence score for SpaCy
        spacy_confidence_score = generate_confidence_score()

        # Measure start time for OpenAI processing
        start_time_openai = time.time()
        
        # Call OpenAI API to generate response
        openai_response = process_with_openai(query)
        
        # Measure end time for OpenAI processing
        end_time_openai = time.time()
        openai_response_time = end_time_openai - start_time_openai  # Calculate time taken for OpenAI
        
        # Measure resource consumption for OpenAI
        openai_cpu_usage, openai_memory_usage = measure_resource_consumption()
        
        # Generate mock confidence score for OpenAI
        openai_confidence_score = generate_confidence_score()

        # Calculate total response time
        total_response_time = time.time() - start_time_spacy

        # Get the 'Bot' user object
        bot_user = User.objects.get(username='Bot')  # Assuming 'Bot' is a valid username for the bot user

        # Save the bot's response to the database with the bot user
        bot_response = ChatMessage.objects.create(user_id=bot_user.id, message=f"{openai_response}")

        # Mock expected response for accuracy calculation
        expected_response = "The author of Harry Potter is J. K. Rowling."

        # Calculate accuracy for both responses
        spacy_accuracy = calculate_accuracy(spacy_response, expected_response)
        openai_accuracy = calculate_accuracy(openai_response, expected_response)

        return JsonResponse({
            'bot_response': bot_response.message, 
            'spacy_response': spacy_response,  
            'spacy_response_time': spacy_response_time,
            'spacy_cpu_usage': spacy_cpu_usage,
            'spacy_memory_usage': spacy_memory_usage,
            'spacy_confidence_score': spacy_confidence_score,
            'spacy_accuracy': spacy_accuracy,
            'openai_response_time': openai_response_time,
            'openai_cpu_usage': openai_cpu_usage,
            'openai_memory_usage': openai_memory_usage,
            'openai_confidence_score': openai_confidence_score,
            'openai_accuracy': openai_accuracy,
            'total_response_time': total_response_time,
            'current_user': current_user.username,
        })

    return render(request, 'chat.html', {
        'bot_response': '', 
        'history': display_history(request.user.id), 
        'current_user': request.user
    })

def process_with_openai(user_input):
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=user_input,
        max_tokens=100
    )
    return response.choices[0].text.strip()

def index(request):
    return render(request, 'index.html')

def login_view(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user:
                login(request, user)
                messages.success(request, 'Login successful.')
                return redirect('chat')
            else:
                messages.error(request, 'Invalid login credentials.')
    else:
        form = LoginForm()

    return render(request, 'login.html', {'form': form})

def register_view(request):
    if request.method == 'POST':
        form = RegistrationForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            confirm_password = form.cleaned_data['confirm_password']

            if password == confirm_password:
                # Create a new user
                user = User.objects.create_user(username=username, password=password)
                
                messages.success(request, 'Registration successful. Please log in.')
                return redirect('login')
            else:
                messages.error(request, 'Password and Confirm Password do not match.')
    else:
        form = RegistrationForm()

    return render(request, 'register.html', {'form': form})

def logout_view(request):
    logout(request)
    messages.success(request, 'Logout successful.')
    return redirect('index')

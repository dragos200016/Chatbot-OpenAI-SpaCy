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
from fuzzywuzzy import fuzz



# Initialize SpaCy with English language model
nlp = spacy.load('en_core_web_sm')

openai.api_key = config.OPENAI_API_KEY



# Define the display_history function
def display_history(user):
    # Retrieve chat history for the user, including bot messages (where user is None)
    history = ChatMessage.objects.filter(user=user) | ChatMessage.objects.filter(user=None)
    return history


def compare_responses(user_input):
    # Call SpaCy API to process user input
    spacy_response = process_with_spacy(user_input)

    # Call OpenAI API to process user input
    openai_response = process_with_openai(user_input)

    # Compare responses
    similarity_score = calculate_similarity(spacy_response, openai_response)

    return spacy_response, openai_response, similarity_score


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

    # Debugging: Print extracted subject, predicate, and object
    print("Subject:", subject)
    print("Predicate:", predicate)
    print("Object:", obj)

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



def calculate_similarity(response1, response2):
    # Calculate similarity between the two responses
    similarity_score = fuzz.ratio(response1.lower(), response2.lower())
    return similarity_score



@login_required
def delete_chat_history(request):
    if request.method == 'POST':
        # Delete chat messages associated with the current user
        ChatMessage.objects.filter(user=request.user).delete()
        return JsonResponse({'success': True})
    else:
        return JsonResponse({'success': False})


def process_user_input(user_input):
    """Process user input with the trained SpaCy model.

    Args:
        user_input (str): The user's input text.

    Returns:
        str: The extracted author name, if found.
    """
    doc = nlp(user_input)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None



@login_required(login_url='login')
def chat(request):
    if request.method == 'POST':
        query = request.POST.get('query', '')
        current_user = request.user

        # Save the user's message to the database
        user_message = ChatMessage.objects.create(user_id=current_user.id, message=query)

        start_time_spacy = time.time()  # Measure start time for SpaCy processing
        
        # Process user input with SpaCy
        spacy_response = process_with_spacy(query)
        
        # Print the SpaCy response
        print("SpaCy Response:", spacy_response)

        end_time_spacy = time.time()  # Measure end time for SpaCy processing
        spacy_response_time = end_time_spacy - start_time_spacy  # Calculate time taken
        
        # Call OpenAI API to generate response
        openai_response = process_with_openai(query)
        
        similarity_score = calculate_similarity(spacy_response, openai_response)

        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.8, system_prompt="You are a machine learning engineer and your job is to answer technical questions."))

        reader = SimpleDirectoryReader(input_dir='./data', recursive=True)
        docs = reader.load_data()

        index = VectorStoreIndex.from_documents(docs, service_context=service_context)

        chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
        response = chat_engine.chat(query)

        end_time = time.time()

        # Get the 'Bot' user object
        bot_user = User.objects.get(username='Bot')  # Assuming 'Bot' is a valid username for the bot user

        # Save the bot's response to the database with the bot user
        bot_response = ChatMessage.objects.create(user_id=bot_user.id, message=f"{response.response}")

        response_time = end_time - start_time_spacy  # Calculate total response time
        print(f"Time taken for SpaCy response: {spacy_response_time} seconds")  # Print SpaCy response time
        print(f"Time taken for GPT-3.5 Turbo response: {response_time} seconds")  # Print total response time
        
        return JsonResponse({
            'bot_response': bot_response.message, 
            'spacy_response': spacy_response,  # Include SpaCy response in the JSON data
            'response_time': response_time, 
            'current_user': current_user.username,
            'similarity_score': similarity_score  # Include similarity score in the JSON data
        })

    return render(request, 'chat.html', {'bot_response': '', 'history': display_history(request.user.id), 'current_user': request.user})

def process_with_openai(user_input):
    # Call the OpenAI API to generate a response based on the user input
    # You can use the OpenAI Python SDK or make HTTP requests directly
    # For example:
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=user_input,
        max_tokens=50
     )
    return response.choices[0].text.strip()

    # For now, let's just return a placeholder response
    # return "This is a placeholder response from OpenAI."

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
                
                # Additional actions if needed
                # For example, you might want to log the user in after registration
                # login(request, user)
                
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
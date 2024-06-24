from .models import ChatMessage
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from .forms import LoginForm, RegistrationForm
from django.http import JsonResponse
from . import config
import openai
import time
import spacy
import psutil
from fuzzywuzzy import fuzz


nlp = spacy.load('en_core_web_sm')

openai.api_key = config.OPENAI_API_KEY

knowledge_base = {
    "author": "J. K. Rowling",
    "characters": [
        "Harry Potter", "Hermione Granger", "Ron Weasley", "Albus Dumbledore",
        "Severus Snape", "Rubeus Hagrid", "Sirius Black", "Draco Malfoy",
        "Neville Longbottom", "Luna Lovegood", "Ginny Weasley", "Fred Weasley",
        "George Weasley", "Minerva McGonagall", "Remus Lupin", "Bellatrix Lestrange",
        "Peter Pettigrew", "Lord Voldemort", "Dobby", "Kreacher", "Cho Chang",
        "Cedric Diggory"
    ],
    "school": "Hogwarts School of Witchcraft and Wizardry",
    "houses": ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"],
    "founders": {
        "Gryffindor": "Godric Gryffindor",
        "Hufflepuff": "Helga Hufflepuff",
        "Ravenclaw": "Rowena Ravenclaw",
        "Slytherin": "Salazar Slytherin"
    },
    "antagonists": ["Lord Voldemort", "Bellatrix Lestrange", "Lucius Malfoy", "Peter Pettigrew"],
    "magical_creatures": [
        "Hippogriffs", "House-elves", "Dementors", "Thestrals", "Dragons",
        "Phoenixes", "Basilisks", "Acromantula", "Merpeople", "Centaurs"
    ],
    "magical_objects": [
        "Elder Wand", "Resurrection Stone", "Invisibility Cloak", "Marauder's Map",
        "Philosopher's Stone", "Horcruxes", "Sorting Hat"
    ],
    "books": [
        "Harry Potter and the Philosopher's Stone", "Harry Potter and the Chamber of Secrets",
        "Harry Potter and the Prisoner of Azkaban", "Harry Potter and the Goblet of Fire",
        "Harry Potter and the Order of the Phoenix", "Harry Potter and the Half-Blood Prince",
        "Harry Potter and the Deathly Hallows"
    ],
    "publishing": {
        "UK": "Bloomsbury",
        "US": "Scholastic Press"
    },
    "genres": ["fantasy", "drama", "coming-of-age fiction", "British school story"],
    "themes": ["prejudice", "corruption", "madness", "death", "friendship", "loyalty", "love", "bravery", "sacrifice"],
    "locations": [
        "Hogwarts", "Diagon Alley", "Hogsmeade", "Forbidden Forest", "Grimmauld Place",
        "Godric's Hollow", "Ministry of Magic", "Azkaban", "The Burrow"
    ],
    "spells": [
        "Expelliarmus", "Avada Kedavra", "Expecto Patronum", "Wingardium Leviosa",
        "Lumos", "Nox", "Accio", "Crucio", "Imperio", "Alohomora", "Stupefy"
    ],
    "quiddich_teams": [
        "Chudley Cannons", "Holyhead Harpies", "Puddlemere United", "Appleby Arrows",
        "Ballycastle Bats", "Falmouth Falcons", "Montrose Magpies", "Wimbourne Wasps"
    ],
    "quiddich_positions": [
        "Seeker", "Keeper", "Beater", "Chaser"
    ],
    "teachers": [
        {"name": "Albus Dumbledore", "subject": "Transfiguration", "position": "Headmaster"},
        {"name": "Minerva McGonagall", "subject": "Transfiguration", "position": "Deputy Headmistress, Head of Gryffindor"},
        {"name": "Severus Snape", "subject": "Potions, Defense Against the Dark Arts", "position": "Head of Slytherin"},
        {"name": "Rubeus Hagrid", "subject": "Care of Magical Creatures", "position": "Keeper of Keys and Grounds"},
        {"name": "Remus Lupin", "subject": "Defense Against the Dark Arts"},
        {"name": "Gilderoy Lockhart", "subject": "Defense Against the Dark Arts"},
        {"name": "Dolores Umbridge", "subject": "Defense Against the Dark Arts", "position": "Hogwarts High Inquisitor"},
        {"name": "Filius Flitwick", "subject": "Charms", "position": "Head of Ravenclaw"},
        {"name": "Pomona Sprout", "subject": "Herbology", "position": "Head of Hufflepuff"},
        {"name": "Sybill Trelawney", "subject": "Divination"},
        {"name": "Horace Slughorn", "subject": "Potions", "position": "Head of Slytherin"},
        {"name": "Quirinus Quirrell", "subject": "Defense Against the Dark Arts"},
        {"name": "Aurora Sinistra", "subject": "Astronomy"},
        {"name": "Bathsheda Babbling", "subject": "Study of Ancient Runes"},
        {"name": "Cuthbert Binns", "subject": "History of Magic"},
        {"name": "Charity Burbage", "subject": "Muggle Studies"},
        {"name": "Rolanda Hooch", "subject": "Flying"}
    ]
}

def display_history(user):
    history = ChatMessage.objects.filter(user=user) | ChatMessage.objects.filter(user=None)
    return history

def process_with_spacy(user_input):
    doc = nlp(user_input)

    subject = None
    predicate = None
    obj = None

    for token in doc:
        if token.dep_ == "nsubj":
            subject = token.text
        elif token.dep_ == "ROOT":
            predicate = token.text
        elif token.dep_ == "dobj":
            obj = " ".join([t.text for t in token.subtree])

    response = "I'm sorry, I don't have information on that topic."

    if doc[0].text.lower() in ["who", "what"]:
        if predicate.lower() in ["write", "wrote"] and obj:
            book = obj.lower()
            if "harry potter" in book:
                author = knowledge_base.get("author", "Unknown")
                response = f"The author of {book.title()} is {author}."
        elif predicate.lower() in ["are", "is"]:
            if "main characters" in user_input.lower():
                characters = ", ".join(knowledge_base["characters"])
                response = f"The main characters in Harry Potter are {characters}."
            elif "school" in user_input.lower():
                school = knowledge_base.get("school", "Unknown")
                response = f"The school in Harry Potter is {school}."
            elif "antagonist" in user_input.lower():
                antagonist = knowledge_base.get("antagonists", ["Unknown"])[0]
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
            elif "teachers" in user_input.lower():
                teachers = ", ".join([teacher["name"] for teacher in knowledge_base["teachers"]])
                response = f"The teachers at Hogwarts include: {teachers}."
            elif "houses" in user_input.lower():
                houses = ", ".join(knowledge_base["houses"])
                response = f"The houses at Hogwarts are: {houses}."
            elif "magical creatures" in user_input.lower():
                magical_creatures = ", ".join(knowledge_base["magical_creatures"])
                response = f"The magical creatures in Harry Potter include: {magical_creatures}."
            elif "magical objects" in user_input.lower():
                magical_objects = ", ".join(knowledge_base["magical_objects"])
                response = f"The magical objects in Harry Potter include: {magical_objects}."
            elif "books" in user_input.lower():
                books = ", ".join(knowledge_base["books"])
                response = f"The Harry Potter series includes the following books: {books}."
            elif "locations" in user_input.lower():
                locations = ", ".join(knowledge_base["locations"])
                response = f"The key locations in Harry Potter include: {locations}."
            elif "spells" in user_input.lower():
                spells = ", ".join(knowledge_base["spells"])
                response = f"Some notable spells in Harry Potter are: {spells}."
            elif "quiddich teams" in user_input.lower():
                quiddich_teams = ", ".join(knowledge_base["quiddich_teams"])
                response = f"The Quidditch teams mentioned in Harry Potter include: {quiddich_teams}."
            elif "quiddich positions" in user_input.lower():
                quiddich_positions = ", ".join(knowledge_base["quiddich_positions"])
                response = f"The positions in Quidditch are: {quiddich_positions}."

    return response


def calculate_accuracy(response, expected_response):
    similarity_ratio = fuzz.token_sort_ratio(response.lower(), expected_response.lower())
    return similarity_ratio / 100.0

def measure_resource_consumption():
    process = psutil.Process()
    cpu_usage = process.cpu_percent(interval=1)
    memory_info = process.memory_info()
    memory_usage = memory_info.rss  
    return cpu_usage, memory_usage

def generate_confidence_score(response, expected_response, response_time, cpu_usage, memory_usage):
    accuracy = calculate_accuracy(response, expected_response)
    response_length = len(response)
    time_factor = 1 / (1 + response_time) 
    resource_factor = 1 / (1 + cpu_usage + memory_usage)

    confidence_score = (accuracy + (response_length / 100) + time_factor + resource_factor) / 4.0

    confidence_score = max(0.5, min(1.0, confidence_score))

    return confidence_score

@login_required
def delete_chat_history(request):
    if request.method == 'POST':
        ChatMessage.objects.filter(user=request.user).delete()
        return JsonResponse({'success': True})
    else:
        return JsonResponse({'success': False})

@login_required(login_url='login')
def chat(request):
    if request.method == 'POST':
        query = request.POST.get('query', '')
        current_user = request.user
        
        expected_response = "The author of Harry Potter is J. K. Rowling."

        user_message = ChatMessage.objects.create(user_id=current_user.id, message=query)

        start_time_spacy = time.time()
        
        spacy_response = process_with_spacy(query)
        
        end_time_spacy = time.time()
        spacy_response_time = end_time_spacy - start_time_spacy  
        
        spacy_cpu_usage, spacy_memory_usage = measure_resource_consumption()
        
        spacy_confidence_score = generate_confidence_score(spacy_response, expected_response, spacy_response_time, spacy_cpu_usage, spacy_memory_usage)


        start_time_openai = time.time()
        
        openai_response = process_with_openai(query)
        
        end_time_openai = time.time()
        openai_response_time = end_time_openai - start_time_openai 
        
        openai_cpu_usage, openai_memory_usage = measure_resource_consumption()
        
        openai_confidence_score = generate_confidence_score(openai_response, expected_response, openai_response_time, openai_cpu_usage, openai_memory_usage)


        total_response_time = time.time() - start_time_spacy

        bot_user = User.objects.get(username='Bot')  # Assuming 'Bot' is a valid username for the bot user

        bot_response = ChatMessage.objects.create(user_id=bot_user.id, message=f"{openai_response}")

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
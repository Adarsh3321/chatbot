import random
import json
import pickle
import numpy as np
import nltk
import subprocess
import webbrowser
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Ensure necessary resources are downloaded
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load intents and model
try:
    with open('C:/Users/LEGION/Desktop/desktop/chatbotpy/intents.json') as f:
        intents = json.load(f)
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    model = load_model('chatbot_model.keras')
except Exception as e:
    print(f"Error loading resources: {e}")
    exit()

def get_response(tag):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm not sure how to respond to that."

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    if results:
        return classes[results[0][0]]  # Return the class with the highest probability
    else:
        return 'noanswer'

def open_application(app_name):
    try:
        if app_name.lower() == "chrome":
            subprocess.Popen(["C:/Program Files/Google/Chrome/Application/chrome.exe"])
        elif app_name.lower() == "notepad":
            subprocess.Popen(["notepad.exe"])
        elif app_name.lower() == "calculator":
            subprocess.Popen(["calc.exe"])
        elif app_name.lower() in ["vscode", "visual studio code"]:
            subprocess.Popen(["C:/Users/LEGION/AppData/Local/Programs/Microsoft VS Code/Code.exe"])
        else:
            return "Sorry, I don't know how to open that application."
        return f"Opening {app_name}..."
    except PermissionError:
        return "Permission denied. Please run the script as an administrator."
    except Exception as e:
        return f"Error opening application: {str(e)}"

def search_web(query):
    url = f"https://www.google.com/search?q={query}"
    webbrowser.open(url)

def handle_message(message):
    if "open" in message.lower():
        app_name = message.lower().replace("open ", "")
        return open_application(app_name)
    elif "search" in message:
        query = message.split("search", 1)[1].strip()
        search_web(query)
        return f"Searching for {query} on Google!"
    else:
        tag = predict_class(message)
        response = get_response(tag)
        return response

print("GO! Bot is running!")

while True:
    message = input("You: ")
    response = handle_message(message)
    print("Bot:", response)

import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from gtts import gTTS
import pygame
import io

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath(r'C:\Users\shaik\Desktop\ChatBot\intents.json')
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Global variable to track audio playback state
is_playing_audio = False

def chatbot(input_text):
    # Input validation
    if not input_text or not isinstance(input_text, str):
        return "Please provide a valid input."
        
    cleaned_input = input_text.strip()
    if not cleaned_input:
        return "Please provide a valid input, not just blank space."
    
    # Check if the input consists only of digits
    if cleaned_input.isdigit():
        return "Please provide a valid input, not just numbers."

    try:
        input_text = vectorizer.transform([cleaned_input])
        tag = clf.predict(input_text)[0]
        
        for intent in intents:
            if intent['tag'] == tag:
                response = random.choice(intent['responses'])
                return response
    except Exception as e:
        return f"An error occurred while processing your request: {e}"

def play_audio(text):
    global is_playing_audio
    if is_playing_audio:
        return  # Prevent playing audio if one is already playing

    is_playing_audio = True  # Set the flag to indicate audio is playing
    tts = gTTS(text=text, lang='en')
    
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)  # Move to the start of the BytesIO object

    pygame.mixer.init()
    pygame.mixer.music.load(audio_fp)
    pygame.mixer.music.play()

    # Wait until the audio is finished playing
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    is_playing_audio = False  # Reset the flag after playback is complete

counter = 0
def main():
    global counter
    st.title("Intents of Chatbot using NLP")

    # Create a sidebar menu with options
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home Menu
    if choice == "Home":
        st.write("Welcome to the chatbot.")

        # Check if the chat_log.csv file exists, and if not, create it with column names
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User   Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:
            # Convert the user input to a string
            user_input_str = str(user_input)

            # Get response only if user input is new
            if 'last_response' not in st.session_state or st.session_state.last_input != user_input_str:
                response = chatbot(user_input)
                st.session_state.last_response = response
                st.session_state.last_input = user_input_str

                # Get the current timestamp
                timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")

                # Save the user input and chatbot response to the chat_log.csv file
                with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow([user_input_str, response, timestamp])

                if response.lower() in ['goodbye', 'bye']:
                    st.write("Thank you for chatting with me. Have a great day!")
                    st.stop()

            # Button to play the response audio
            if st.button("Read Response"):
                if 'last_response' in st.session_state:
                    play_audio(st.session_state.last_response)

        # Ensure the last response is displayed even after clicking the button
        if 'last_response' in st.session_state:
            st.text_area("Chatbot:", value=st.session_state.last_response, height=120, max_chars=None, key="chatbot_response")

    # Conversation History Menu
    elif choice == "Conversation History":
        st.header("Conversation History")
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip the header row
            for row in csv_reader:
                st.markdown(f"**User    :** {row[0]}")
                st.markdown(f"**Chatbot:** {row[1]}")
                st.markdown(f"**Timestamp:** {row[2]}")
                st.markdown("---")
        # Clear chat history button
        if st.button("Clear Chat History"):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User   Input', 'Chatbot Response', 'Timestamp'])
            st.success("Chat history cleared!")

    elif choice == "About":
        st.write("The goal of this project is to create a chatbot that can understand and respond to user input based on intents. The chatbot is built using Natural Language Processing (NLP) library and Logistic Regression, to extract the intents and entities from user input. The chatbot is built using Streamlit, a Python library for building interactive web applications.")

        st.subheader("Project Overview:")
        st.write("""
        The project is divided into two parts:
        1. NLP techniques and Logistic Regression algorithm is used to train the chatbot on labeled intents and entities.
        2. For building the Chatbot interface, Streamlit web framework is used to build a web-based chatbot interface. The interface allows users to input text and receive responses from the chatbot.
        """)

        st.subheader("Dataset:")
        st.write("""
        The dataset used in this project is a collection of labelled intents and entities. The data is stored in a list.
        - Intents: The intent of the user input (e.g. "greeting", "budget", "about")
        - Entities: The entities extracted from user input (e.g. "Hi", "How do I create a budget?", "What is your purpose?")
        - Text: The user input text.
        """)

        st.subheader("Streamlit Chatbot Interface:")
        st.write("The chatbot interface is built using Streamlit. The interface includes a text input box for users to input their text and a chat window to display the chatbot's responses. The interface uses the trained model to generate responses to user input.")

        st.subheader("Conclusion:")
        st.write("In this project, a chatbot is built that can understand and respond to user input based on intents. The chatbot was trained using NLP and Logistic Regression, and the interface was built using Streamlit. This project can be extended by adding more data, using more sophisticated NLP techniques, deep learning algorithms.")

if __name__ == '__main__':
    main()
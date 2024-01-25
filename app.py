import streamlit as st
import speech_recognition as sr
from openai import OpenAI
import os
from pathlib import Path
from playsound import playsound
import threading
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.playback import play

# Load environment variables from .env
load_dotenv()

client = OpenAI(api_key=os.environ['openai_api_key'])

# Initialize the recognizer
r = sr.Recognizer()
sample_rate = 44100
duration = 5
r.energy_threshold = 4000
r.dynamic_energy_threshold = True
r.dynamic_energy_adjustment_damping = 0.15
r.dynamic_energy_ratio = 1.5

st.title('Rehan AI Voice Assistant')

def listen():
    listening_message = st.toast("Listening...")
    try:    
        with sr.Microphone() as source:
            audio_data = r.listen(source, timeout=duration, phrase_time_limit=duration)

        audio_path = 'audio.wav'
        with open(audio_path, 'wb') as audio_file:
            audio_file.write(audio_data.get_wav_data())

        with sr.AudioFile(audio_path) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data, key=None, language="en-US", show_all=False)
            print("you said:", text)

            listening_message.toast("Listening Completed.... ")

            # Recognize speech using Google's speech recognition
            transcripted = r.recognize_google(audio_data)

            # Display transcription
            st.warning(transcripted)

            # Make API call to OpenAI for generating response
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant of Rehan Ai. who helps everyone with their queries and gives the right answer "},
                    {"role": "user", "content": "Who are you?"},
                    {"role": "assistant", "content": "As an AI language model, I am programmed to assist you with your queries and concerns to the best of my abilities"},
                    {"role": "user", "content": "Where was it?"},
                    {"role": "user", "content": f"This is the transcribed text: {transcripted}"}
                ]
            )

            # Get AI response content
            response_content = response.choices[0].message.content

            # Display AI response
            st.success(response_content)

            speech_file_path = Path(__file__).parent / "output.mp3"
            audio_response = client.audio.speech.create(
                model="tts-1",
                voice="echo",
                input=response_content
            )
            audio_response.stream_to_file(speech_file_path)

            # Play the generated audio in a separate thread
            threading.Thread(target=playsound, args=(str(speech_file_path),)).start()

            # Generate audio file from AI response
            # speech = AudioSegment.from_text(response_content, voice="en", lang="en")
            # speech.export("output.mp3", format="mp3")

            # # Play the generated audio
            # threading.Thread(target=play, args=("output.mp3",)).start()

    except sr.WaitTimeoutError:
        st.error('No speech detected within the timeout period.')
    except sr.UnknownValueError:
        st.error('Could not understand audio')
    except sr.RequestError as e:
        st.error(f'Request to Google API failed: {e}')
    except Exception as e:
        st.error(f'An unexpected error occurred: {e}')

# UI components
if st.button("🎤"):
    # Call the listen function when the button is clicked
    listen()

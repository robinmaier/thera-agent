import os
import json
import time
import sounddevice as sd
import soundfile as sf
from datetime import datetime
from pathlib import Path
import numpy as np
from openai import OpenAI
import subprocess

class ConversationApp:
    def __init__(self):
        self.client = OpenAI()
        self.sample_rate = 44100
        self.conversations_dir = Path("conversations")
        self.conversations_dir.mkdir(exist_ok=True)
        
    def record_audio(self):
        print("\nPress Enter to start your recording...")
        input()
        print("Recording... Press Enter to stop.")
        
        audio_data = []
        recording = True
        
        def callback(indata, frames, time, status):
            if recording:
                audio_data.append(indata.copy())
        
        stream = sd.InputStream(samplerate=self.sample_rate, channels=1, callback=callback)
        with stream:
            input()
            recording = False
            
        audio = np.concatenate(audio_data, axis=0)
        temp_file = "temp_recording.mp3"
        sf.write(temp_file, audio, self.sample_rate)
        
        return temp_file

    def speech_to_text(self, audio_file):
        with open(audio_file, "rb") as file:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=file
            )
        return transcript.text

    def text_to_speech(self, text):
        response = self.client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        
        temp_audio = "temp_speech.mp3"
        response.stream_to_file(temp_audio)
        
        subprocess.run(['mpg123', temp_audio], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def generate_response(self, messages):
        completion = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        return completion.choices[0].message.content

    def load_conversation_history(self):
        history = []
        for file in self.conversations_dir.glob("conversation_*.json"):
            with open(file, "r") as f:
                history.extend(json.load(f))
        return history

    def save_conversation(self, conversation):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{timestamp}.json"
        with open(self.conversations_dir / filename, "w") as f:
            json.dump(conversation, f, indent=2)

    def run(self):
        conversation = []
        history = self.load_conversation_history()
        
        if not history:
            # First time conversation
            initial_message = "Hallo, ich bin Lou. Wie heißt du?"
            print("\nAssistant:", initial_message)
            self.text_to_speech(initial_message)
            
            audio_file = self.record_audio()
            user_response = self.speech_to_text(audio_file)
            user_name = user_response.split("ich bin ")[-1].strip()
            
            next_message = f"Hi {user_name}. Über was möchtest du mit mir sprechen?"
            print("\nAssistant:", next_message)
            self.text_to_speech(next_message)
            
            conversation.extend([
                {"role": "assistant", "content": initial_message},
                {"role": "user", "content": user_response},
                {"role": "assistant", "content": next_message}
            ])
        
        else:
            # Returning user conversation
            system_prompt = f"""Versetze dich in die Rolle eines Psychotherapeuten und lese die bisherige Konversation {history}. 
            Wähle ein Thema, welches dir relevant erscheint und über das du in deiner Rolle als Psychotherapeut sprechen möchtest. 
            Finde außerdem den Namen des Users und beginne das Gespräch wie folgt.
            
            "Hallo {'{Name des Users}'}. Willkommen zurück. Möchtest du heute über {'{Thema}'} sprechen?"
            """
            
            messages = [{"role": "system", "content": system_prompt}]
            initial_message = self.generate_response(messages)
            print("\nAssistant:", initial_message)
            self.text_to_speech(initial_message)
            conversation.append({"role": "assistant", "content": initial_message})

        while True:
            print("\nPress Enter to start conversation or type 'exit' to end conversation:")
            user_input = input()
            
            if user_input.lower() == 'exit':
                break
                
            audio_file = self.record_audio()
            user_response = self.speech_to_text(audio_file)
            print("User:", user_response)
            conversation.append({"role": "user", "content": user_response})
            
            messages = [
                {"role": "system", "content": "Versetze dich in die Rolle eines Psychotherapeuten und reagiere mit einer kurzen Antwort, um das Gespräch fortzuführen. Spreche den User immer mit 'du' an."},
                *[{"role": m["role"], "content": m["content"]} for m in conversation]
            ]
            
            assistant_response = self.generate_response(messages)
            print("\nAssistant:", assistant_response)
            self.text_to_speech(assistant_response)
            conversation.append({"role": "assistant", "content": assistant_response})

        self.save_conversation(conversation)
        print("\nConversation saved. Goodbye!")

if __name__ == "__main__":
    app = ConversationApp()
    app.run()

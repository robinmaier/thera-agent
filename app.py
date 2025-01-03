import os
from dotenv import load_dotenv
import json
import time
import sounddevice as sd
import soundfile as sf
from datetime import datetime
from pathlib import Path
import numpy as np
from openai import OpenAI
import subprocess

# Lade Umgebungsvariablen aus .env Datei
load_dotenv()

class ConversationApp:
    def __init__(self):

        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            project=os.getenv("OPENAI_PROJECT_ID")
        )
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
        
        with open(temp_audio, 'wb') as file:
            for chunk in response.iter_bytes():
                file.write(chunk)
        
        subprocess.run(['mpg123', temp_audio], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def generate_response(self, messages):
        completion = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        return completion.choices[0].message.content

    def load_conversation_history(self):
        history = []
        
        # Sammle alle Dateien mit ihren Timestamps
        files_with_timestamps = []
        for file in self.conversations_dir.glob("conversation_*.json"):
            try:
                # Extrahiere Timestamp aus dem Dateinamen (z.B. "conversation_20240301_123456.json")
                timestamp = file.stem.split('_')[1] + '_' + file.stem.split('_')[2]
                files_with_timestamps.append((file, timestamp))
            except IndexError:
                print(f"Warnung: Ungültiges Dateiformat: {file}")
                continue
        
        # Sortiere nach Timestamp
        files_with_timestamps.sort(key=lambda x: x[1])
        
        # Lade Dateien in sortierter Reihenfolge
        for file, _ in files_with_timestamps:
            with open(file, "r") as f:
                history.extend(json.load(f))
        
        return history

    def save_conversation(self, conversation_data):
        filename = f"conversation_{conversation_data['metadata']['start_time']}.json"
        
        with open(self.conversations_dir / filename, "w", encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)

    def generate_summary(self, conversation):
        messages = [
            {
                "role": "system", 
                "content": """Versetze dich in die Rolle eines Psychotherapeuten. 
                Die nachfolgende conversation-Liste enthält ein vollständiges Gespräch zwischen dir und dem User.
                
                Deine Aufgabe:
                1. Fasse dieses eine Gespräch kurz zusammen
                2. Interpretiere den Gesprächsverlauf aus therapeutischer Sicht
                3. Spreche von "wir" und sprich den User mit "du" an
                4. Stelle keine Fragen

                Hier ist ein vollständiges Beispiel für eine Zusammenfassung:
                "In unserem Gespräch hast du den Wunsch geäußert, über Verlustängste und die Angst 
                vor dem Alleinsein zu sprechen. Wir haben gemeinsam erkundet, wie diese Ängste 
                dein tägliches Leben beeinflussen. Aus therapeutischer Sicht zeigt sich, dass 
                deine Befürchtungen eng mit früheren Erfahrungen verbunden sind. Deine Offenheit 
                im Gespräch deutet auf eine hohe Bereitschaft zur Selbstreflexion hin."
                """
            },
            *[{"role": m["role"], "content": m["content"]} for m in conversation]
        ]
        
        summary = self.generate_response(messages)
        return summary

    def save_summary(self, summary, timestamp):
        summary_dir = Path("conversation_summary")
        summary_dir.mkdir(exist_ok=True)
        
        filename = f"summary_{timestamp}.json"
        with open(summary_dir / filename, "w") as f:
            json.dump({"summary": summary}, f, indent=2, ensure_ascii=False)

    def run(self):
        # Erfasse Start-Timestamp
        start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        
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
            
            next_message = f"Hi {user_name}. Über was möchtest du mit mir sprechen? Wir können über alles sprechen, was dich beschäftigt. Beziehungen, Arbeit, Ziele, Konflikte, etc."
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
            Finde außerdem den Namen des Users und beginne das Gespräch wie folgt:
            - Begrüße den User mit seinem Namen
            - Starte das Gespräch mit einer kurzen Frage, in die du das Thema einbaust, welches dir relevant erscheint
            - Stelle sicher, dass es nur um ein Thema geht
            - Halte dich kurz und spreche nicht zu lange

            Hier ein Beispiel für eine kurze prägnante Gesprächseröffnung:
            "Hallo Robin. Ich freue mich, dass du heute hier bist. Wie geht es dir mit der Trennung von deiner Partnerin?"
            """
            
            messages = [{"role": "system", "content": system_prompt}]
            initial_message = self.generate_response(messages)
            print("\nAssistant:", initial_message)
            self.text_to_speech(initial_message)
            conversation.append({"role": "assistant", "content": initial_message})

        while True:
            print("\nPress Enter to continue conversation or type 'exit' to end conversation:")
            user_input = input()
            
            if user_input.lower() == 'exit':
                break
                
            audio_file = self.record_audio()
            user_response = self.speech_to_text(audio_file)
            print("User:", user_response)
            conversation.append({"role": "user", "content": user_response})
            
            messages = [
                {"role": "system", "content": """Versetze dich in die Rolle eines Psychotherapeuten.
                
                Gesprächsführung:
                1. Reagiere mit kurzen, empathischen Antworten
                2. Spreche den User immer mit 'du' an

                Abschluss eines Themas, wenn:
                - Der User keine weiteren Gedanken zum Thema hat
                - Der User explizit über etwas anderes sprechen möchte
                
                Reagiere auf den Gesprächsabschluss wie folgt:
                - Frage den User, ob er weiter über das Thema sprechen möchte, aber nur, wenn er das nicht bereits explizit gesagt hat
                - Falls nicht, biete dem User an, über etwas anderes zu sprechen und führe das Gespräch fort
                """},
                *[{"role": m["role"], "content": m["content"]} for m in conversation]
            ]
            
            assistant_response = self.generate_response(messages)
            print("\nAssistant:", assistant_response)
            self.text_to_speech(assistant_response)
            conversation.append({"role": "assistant", "content": assistant_response})

        # Erfasse End-Timestamp
        end_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Erstelle Metadaten für die Konversation
        conversation_data = {
            "metadata": {
                "start_time": start_time,
                "end_time": end_time
            },
            "messages": conversation
        }
        
        # Speichere Konversation mit Metadaten
        self.save_conversation(conversation_data)
        
        # Generiere und speichere Zusammenfassung
        summary = self.generate_summary(conversation)
        self.save_summary(summary, end_time)  # Verwende end_time als Timestamp
        
        print("\nConversation and summary saved. Goodbye!")

if __name__ == "__main__":
    app = ConversationApp()
    app.run()

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
    
    # Sortiere Dateien nach Timestamp im Dateinamen
        files = sorted(
            self.conversations_dir.glob("conversation_*.json"),
            key=lambda x: x.stem.split('_')[1])
    
        for file in files:
            with open(file, "r", encoding='utf-8') as f:
                history.extend(json.load(f))
    
        return history
        print (history)

    def save_conversation(self, conversation, timestamp):
        filename = f"conversation_{timestamp}.json"
    
        with open(self.conversations_dir / filename, "w", encoding='utf-8') as f:
            json.dump(conversation, f, indent=2, ensure_ascii=False)

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
            
            next_message = f"Hi {user_name}. Gibt es was, über das du mit mir sprechen möchtest?"
            print("\nAssistant:", next_message)
            self.text_to_speech(next_message)
            
            conversation.extend([
                {"role": "assistant", "content": initial_message},
                {"role": "user", "content": user_response},
                {"role": "assistant", "content": next_message}
            ])
        
        else:
            # Returning user conversation
            messages = [
                {
                    "role": "system", 
                    "content":"""Versetze dich in die Rolle eines Psychotherapeuten.

                    WICHTIG: Nach diesem System-Prompt folgt eine Liste von chronologisch sortierten Nachrichten 
                    aus vorherigen Gesprächen. Diese Nachrichten dienen nur als Kontext.
                
                    Deine Aufgabe:
                    1. Lies die bisherige Konversationen
                    2. Finde den Namen des Users
                    3. Identifiziere wichtige Themen aus den vorherigen Gesprächen
                   
                    Dann beginne ein NEUES Gespräch wie folgt:
                    - Wähle EIN relevantes Thema aus den vorherigen Gesprächen 
                    - Begrüße den User mit seinem Namen
                    - Formuliere eine kurze, prägnante Eröffnungsfrage, in die du das zuvor ausgewählte Thema einbaust

                    Beispiel:
                    "Hallo Robin. Ich freue mich, dass du heute hier bist. Das letzte mal haben wir über die Trennung 
                    von deiner Partnerin gesprochen. Wie geht es dir damit?"
                    """ 
                 },
                 *[{"role": m["role"], "content": m["content"]} for m in history]
            ]
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
                {
                    "role": "system", 
                    "content": """Versetze dich in die Rolle eines Psychotherapeuten. Die Ziele für das
                    Gespräch sind:

                    Selbstreflexion und Selbsterkenntnis fördern
                    - Der User soll sich seinen Gedanken, Gefühlen und Verhaltensmustern bewusst werden.
                    - Der User soll Zusammenhänge zwischen aktuellen Problemen und möglichen Ursachen erkennen.

                    Problemlösungsstrategien entwickeln
                    - Neue Ansätze erarbeiten, um Herausforderungen im Alltag zu bewältigen.
                    - Werkzeuge an die Hand bekommen, um Stress, Ängste oder Konflikte besser zu bewältigen.

                    Emotionale Entlastung
                    - Raum schaffen, um Gefühle in einer sicheren, wertfreien Umgebung auszudrücken.
                    - Emotionale Stabilität fördern.

                    Änderung dysfunktionaler Muster
                    - Ungünstige Denk- und Verhaltensmuster identifizieren und durch hilfreichere ersetzen.
                    - Negative Glaubenssätze oder Automatismen hinterfragen.

                    Förderung von Ressourcen und Resilienz
                    - Persönliche Stärken und Ressourcen entdecken und aktivieren.
                    - Die Fähigkeit entwickeln, mit schwierigen Situationen besser umzugehen.

                    Beziehungs- und Kommunikationsfähigkeiten verbessern
                    - Zwischenmenschliche Dynamiken reflektieren und optimieren.
                    - Gesunde Grenzen setzen lernen und bessere Kommunikationsmuster entwickeln.
                
                    Gesprächsführung:
                    1. Reagiere mit kurzen, empathischen Antworten
                    2. Spreche den User immer mit 'du' an

                    Abschluss eines Themas, wenn:
                    - Der User keine weiteren Gedanken zum Thema hat
                    - Der User explizit über etwas anderes sprechen möchte
                
                    Reagiere auf den Gesprächsabschluss wie folgt:
                    - Frage den User, ob er weiter über das Thema sprechen möchte, aber nur, wenn er das nicht bereits explizit gesagt hat
                    - Falls nicht, biete dem User an, über etwas anderes zu sprechen und führe das Gespräch fort
                    """
                },
                *[{"role": m["role"], "content": m["content"]} for m in conversation]
            ]
            
            assistant_response = self.generate_response(messages)
            print("\nAssistant:", assistant_response)
            self.text_to_speech(assistant_response)
            conversation.append({"role": "assistant", "content": assistant_response})

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Speichere Konversation
        self.save_conversation(conversation, timestamp)
        
        # Generiere und speichere Zusammenfassung
        summary = self.generate_summary(conversation)
        self.save_summary(summary, timestamp)
        
        print("\nConversation and summary saved. Goodbye!")

if __name__ == "__main__":
    app = ConversationApp()
    app.run()

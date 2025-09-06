import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import tempfile
import os
import queue
import time
import requests
import subprocess
import platform
import psutil
import shutil
import threading
import re
import unicodedata
import random
import pyautogui
from langdetect import detect
from ollama import chat
import importlib
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import json

"""
Intégration :
- Détection floue des commandes vocales ("ajoute un outil", "prendre des notes", "au revoir", …)
  via un mapping alphabet -> nombres + comparaison en pourcentage.
- Messages aléatoires (tirage_au_sort) et capture d’écran optionnelle pour les matches ≥ seuil.

Ajustez les constantes dans la section "--- Paramètres ---" ci‑dessous selon vos besoins.
"""

# --- Paramètres ---
SAMPLE_RATE = 16000
CHANNELS = 1
TRIGGER_WORD = "hello"
SILENCE_THRESHOLD_DB = -35
SILENCE_DURATION = 1
FRAME_DURATION = 0.5
OLLAMA_MODEL = "gemma3:1b"
CODESRAL_MODEL = "codestral:latest"
JARVIS_FILE = "JARVIS2.5.py"

# Seuil de correspondance (%) pour considérer qu’une commande est reconnue
FUZZY_THRESHOLD = 70.0
# Si True, enregistre une capture d’écran quand une commande dépasse SCREENSHOT_PCT_MIN
ENABLE_SCREENSHOT_ON_MATCH = False
SCREENSHOT_PCT_MIN = 30.0

PERSONALITY_PROMPT = (
    "Tu es JARVIS, mon assistant personnel. ",
    "Je m appelles Simon, et tu es mon assistant personnel. ",
    "Tu es une IA conçue pour m'aider dans ma vie quotidienne, et pour te servir de mon ordinateur via des outils. ",
    "Tu réponds de manière simple et efficace. ",
    "Tu es capable de comprendre et d'exécuter des commandes vocales, de prendre des notes, ou de coder des outils dynamiques grâce à codestral. ",
    "Tu te sers de ta mémoire uniquement lorsque je te le demande, sinon tu ne l'utilises pas pour répondre."
)

q = queue.Queue()
say_process = None
history = []
audio_amplitude = 0

# --- Initialisation Flask ---
app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print("✅ Client WebSocket connecté")
    emit('status', {'message': 'Connecté au serveur JARVIS'})

@socketio.on('audio_amplitude')
def handle_amplitude(data):
    global audio_amplitude
    audio_amplitude = data['amplitude']
    print(f"📈 Amplitude reçue: {audio_amplitude}")

# --- Tirage aléatoire (messages feedback) ---
def tirage_au_sort():
    messages = [
        "Bonjour !",
        "😅 Désolé, essaie encore...",
        "🎉 Bravo, super tirage !",
        "🙃 Pas de chance cette fois",
        "🔥 Jackpot !"
    ]
    return random.choice(messages)

# --- Fuzzy matching basé alphabet -> nombres ---
# dictionnaire alphabet (a=1, b=2, ... z=26)
alphabet = {chr(i + 96): i for i in range(1, 27)}

def strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def normalize_text(texte: str) -> str:
    # minuscules, supprime accents, garde lettres et espaces simples
    t = strip_accents(texte.lower())
    t = re.sub(r"[^a-z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def mot_to_numbers(texte: str):
    """Convertit un texte en liste de nombres (ignore espaces et caractères non alphabétiques)."""
    return [alphabet[lettre] for lettre in normalize_text(texte) if lettre in alphabet]

def comparer_suites(ref, test):
    """Compare deux suites et calcule le pourcentage de correspondance + erreurs."""
    if len(ref) != len(test):
        print("⚠️ Attention : longueurs différentes.")
    total = min(len(ref), len(test)) if len(ref) and len(test) else 0
    corrects = sum(1 for r, t in zip(ref, test) if r == t)
    faux = total - corrects
    pourcentage_correct = (corrects / len(ref) * 100) if len(ref) else 0.0
    pourcentage_faux = 100 - pourcentage_correct
    erreurs = [(i, r, t) for i, (r, t) in enumerate(zip(ref, test)) if r != t]
    return pourcentage_correct, pourcentage_faux, erreurs

# Commandes cibles et variantes acceptées
COMMAND_PHRASES = {
    "add_tool": [
        "ajoute un outil",
        "ajouter un outil",
        "ajoute outil",
        "ajoute une fonction",
        "ajouter une fonction"
    ],
    "take_notes": [
        "prendre des notes",
        "prends des notes",
        "ecris une note",
        "note ceci",
        "prend des notes"
    ],
    "goodbye": [
        "au revoir",
        "a bientot",
        "arrete toi",
        "stoppe la",
        "fin de la conversation"
    ]
}

# Pré-calcul des suites de référence
COMMAND_REFS = {
    key: [mot_to_numbers(phrase) for phrase in phrases]
    for key, phrases in COMMAND_PHRASES.items()
}

def best_matching_command(text: str):
    """Retourne (cmd_key, best_pct, best_phrase, erreurs) ou (None, 0, None, [])"""
    text_norm = normalize_text(text)
    suite_test = mot_to_numbers(text_norm)
    best = (None, 0.0, None, [])
    for key, ref_list in COMMAND_REFS.items():
        for phrase, suite_ref in zip(COMMAND_PHRASES[key], ref_list):
            pc_ok, pc_faux, erreurs = comparer_suites(suite_ref, suite_test)
            if pc_ok > best[1]:
                best = (key, pc_ok, phrase, erreurs)
    return best

# --- Fonctions système ---
def start_ollama_model_if_needed():
    for proc in psutil.process_iter(['name', 'cmdline']):
        try:
            if proc.info['name'] and 'ollama' in proc.info['name'].lower() and 'run' in ' '.join(proc.info.get('cmdline') or []):
                print("✅ Ollama est déjà en cours d'exécution.")
                return
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    print("🚀 Lancement de Ollama avec le modèle gemma3:1b...")
    if platform.system() == "Darwin":
        ollama_path = shutil.which("ollama")
        if not ollama_path:
            print("❌ Ollama introuvable.")
            return
        apple_script = f'''
        tell application "Terminal"
            do script "{ollama_path} run {OLLAMA_MODEL}"
        end tell
        '''
        subprocess.run(["osascript", "-e", apple_script])
    else:
        print("❌ Fonction non compatible hors macOS.")

# --- Audio utils ---
def rms_db(audio):
    global audio_amplitude
    rms = np.sqrt(np.mean(audio ** 2))
    audio_amplitude = float(rms)
    socketio.emit('amplitude_update', {'amplitude': audio_amplitude})
    return -100 if rms == 0 else 20 * np.log10(rms)

def save_audio(audio, sample_rate):
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wavfile.write(temp_wav.name, sample_rate, audio)
    return temp_wav.name

def transcribe_audio(file_path):
    model = whisper.load_model("base")
    print("🧠 Transcription en cours...")
    result = model.transcribe(file_path)
    socketio.emit('transcription', {'text': result["text"]})
    return result["text"]

def detect_trigger_word():
    print("🕵️ Dis 'hello'...")
    audio = sd.rec(int(3 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    temp_wav = save_audio(audio, SAMPLE_RATE)
    text = transcribe_audio(temp_wav)
    os.remove(temp_wav)
    return TRIGGER_WORD in text.lower()

def audio_callback(indata, frames, time_info, status):
    if status:
        print("⚠️", status)
    q.put(indata.copy())

def dynamic_record_stream():
    print("🎙️ Enregistrement lancé...")
    recorded = []
    silence_time = 0
    frame_size = int(FRAME_DURATION * SAMPLE_RATE)
    with sd.InputStream(callback=audio_callback, channels=CHANNELS, samplerate=SAMPLE_RATE, blocksize=frame_size):
        while True:
            try:
                frame = q.get(timeout=1)
            except queue.Empty:
                print("❌ Timeout lecture micro")
                break
            frame = frame.flatten()
            recorded.append(frame)
            db = rms_db(frame)
            silence_time = silence_time + FRAME_DURATION if db < SILENCE_THRESHOLD_DB else 0
            if silence_time > SILENCE_DURATION:
                print("🤫 Silence détecté.")
                break
    return np.concatenate(recorded) if recorded else np.array([], dtype=np.float32)

# --- IA avec mémoire ---
def send_to_ollama(prompt, model=OLLAMA_MODEL, history=None):
    print("💬 Envoi à Ollama...")
    context = "".join(f"Utilisateur : {u}\nAssistant : {a}\n" for u, a in history) if history else ""
    full_prompt = f"System : {''.join(PERSONALITY_PROMPT)}\n{context}Utilisateur : {prompt}\nAssistant :"
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            headers={"Content-Type": "application/json"},
            json={"model": model, "prompt": full_prompt, "stream": False}
        )
        if response.status_code == 200:
            response_text = response.json().get("response", "")
            socketio.emit('response', {'text': response_text})
            return response_text
        else:
            error = f"Erreur Ollama : {response.text}"
            socketio.emit('error', {'message': error})
            return error
    except Exception as e:
        error = f"Erreur connexion Ollama : {e}"
        socketio.emit('error', {'message': error})
        return error

# --- Synthèse vocale ---
def interruptible_speak(text):
    global say_process
    try:
        language = detect(text)
        voice = "Thomas" if language == "fr" else "Samantha"
        say_process = subprocess.Popen(["say", "-v", voice, text])

        def listen_for_interrupt():
            with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=1024) as stream:
                while say_process.poll() is None:
                    audio = stream.read(1024)[0].flatten()
                    if rms_db(audio) > SILENCE_THRESHOLD_DB + 10:
                        print("🛑 Interruption détectée !")
                        say_process.terminate()
                        break

        listener_thread = threading.Thread(target=listen_for_interrupt)
        listener_thread.start()
        say_process.wait()
        listener_thread.join()
    except Exception as e:
        print(f"Erreur synthèse vocale : {e}")

# --- Import dynamique ---
def import_lumen_functions():
    try:
        spec = importlib.util.spec_from_file_location("JARVIS", JARVIS_FILE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore
        globals().update({k: getattr(module, k) for k in dir(module) if not k.startswith("__")})
        print("✅ Fonctions JARVIS importées.")
    except Exception as e:
        print("❌ Erreur import fonctions JARVIS.py :", e)

# --- Début outils dynamiques ---
def valider_code(code):
    try:
        compile(code, "<string>", "exec")
        return True
    except SyntaxError as e:
        print("⚠️ Code invalide :", e)
        return False

def coder_fonction_avec_codestral(demande):
    start_ollama_model_if_needed()
    path = os.path.realpath(__file__)
    try:
        with open(path, "r", encoding="utf-8") as f:
            main_code = f.read()
    except Exception as e:
        return None
    system_msg = (
        "Tu es un assistant Python. À partir du script principal d’un assistant vocal, génère UNE SEULE fonction complète à insérer dans JARVIS.py. "
        "Ta réponse commence par 'def', sans explication ni commentaire.\n\n"
        f"Code principal :\n{main_code}"
    )
    try:
        response = chat(model=CODESRAL_MODEL, messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": demande}
        ]).message.content.strip()
        if not response.startswith("def ") or not valider_code(response):
            print("⚠️ Mauvais code généré.")
            return None
        return response
    except Exception as e:
        print("Erreur Codestral :", e)
        return None

def insert_code_in_script(nom_fonction, code_fonction):
    try:
        if not os.path.exists(JARVIS_FILE):
            with open(JARVIS_FILE, "w") as f:
                f.write("# --- Début outils dynamiques ---\n# --- Fin outils dynamiques ---\n")
        with open(JARVIS_FILE, "r", encoding="utf-8") as f:
            content = f.read()
        if nom_fonction in content:
            print(f"🔁 Fonction '{nom_fonction}' déjà présente.")
            return
        nouveau_code = f"\n# --- Fonction ajoutée dynamiquement ---\n{code_fonction}\n"
        content = re.sub(r"(# --- Début outils dynamiques ---\n)(.*?)(# --- Fin outils dynamiques ---)",
                         lambda m: m.group(1) + m.group(2) + nouveau_code + m.group(3),
                         content, flags=re.DOTALL)
        with open(JARVIS_FILE, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ Fonction '{nom_fonction}' ajoutée dans {JARVIS_FILE}")
    except Exception as e:
        print("Erreur insertion :", e)

def ajouter_outil_vocal():
    interruptible_speak("Quel outil veux-tu que je code ?")
    audio = dynamic_record_stream()
    if audio.size == 0:
        interruptible_speak("Je n'ai rien entendu.")
        return
    demande = transcribe_audio(save_audio(audio, SAMPLE_RATE))
    code = coder_fonction_avec_codestral(demande)
    if code:
        match = re.search(r"def (\w+)\(", code)
        if match:
            nom_fonction = match.group(1)
            insert_code_in_script(nom_fonction, code)
            try:
                exec(code, globals())
                import_lumen_functions()
                if nom_fonction in globals():
                    interruptible_speak("Outil ajouté. Je l’exécute.")
                    globals()[nom_fonction]()
                else:
                    interruptible_speak("Erreur de chargement.")
            except Exception as e:
                interruptible_speak("Erreur pendant l’exécution.")
                print(e)
        else:
            interruptible_speak("Je n'ai pas compris le nom de la fonction.")
    else:
        interruptible_speak("Erreur pendant la génération du code.")

# --- Notes ---
def prendre_des_notes():
    print("📝 Dictée de note en cours...")
    interruptible_speak("Ok, j'ouvre Notes. Que veux-tu que je note ?")
    audio = dynamic_record_stream()
    if audio.size == 0:
        interruptible_speak("Je n'ai rien entendu.")
        return
    wav_path = save_audio(audio, SAMPLE_RATE)
    texte_note = transcribe_audio(wav_path)
    os.remove(wav_path)
    interruptible_speak("J'écris ta note.")
    apple_script = f'''
    tell application "Notes"
        activate
        tell account "iCloud"
            make new note at folder "Notes" with properties {{name:"Note dictée", body:"{texte_note}"}}
        end tell
    end tell
    '''
    subprocess.run(["osascript", "-e", apple_script])

# --- Boucle principale JARVIS ---
def run_jarvis():
    start_ollama_model_if_needed()
    import_lumen_functions()
    in_conversation = False
    while True:
        if not in_conversation:
            if detect_trigger_word():
                in_conversation = True
                interruptible_speak("Bonjour, je t'écoute.")
                socketio.emit('status', {'message': 'JARVIS activé'})
            else:
                print("❌ 'hello' non détecté.")
                continue

        audio = dynamic_record_stream()
        if audio.size == 0:
            print("🤫 Rien capté, retour veille.")
            in_conversation = False
            socketio.emit('status', {'message': 'Mode veille'})
            continue

        wav_path = save_audio(audio, SAMPLE_RATE)
        transcription = transcribe_audio(wav_path)
        os.remove(wav_path)

        print("\n📄 Transcription :")
        print(transcription)
        transcription_lower = transcription.lower().strip()
        if not transcription_lower:
            print("🤫 Transcription vide, passage en mode veille.")
            in_conversation = False
            socketio.emit('status', {'message': 'Mode veille'})
            continue

        # --- Détection floue de commande ---
        cmd_key, pc_ok, phrase_ref, erreurs = best_matching_command(transcription_lower)
        print(f"🔎 Meilleure commande: {cmd_key} ({pc_ok:.2f}%) vs '{phrase_ref}'")

        if ENABLE_SCREENSHOT_ON_MATCH and pc_ok >= SCREENSHOT_PCT_MIN:
            try:
                pyautogui.screenshot("capture.png")
                print("📸 Capture d’écran enregistrée sous 'capture.png'")
            except Exception as e:
                print("⚠️ Capture écran impossible:", e)

        if cmd_key and pc_ok >= FUZZY_THRESHOLD:
            # Petit feedback sympa
            feedback = tirage_au_sort()
            print(feedback)
            socketio.emit('status', {'message': f'Commande reconnue: {phrase_ref} ({pc_ok:.1f}%)'})

            if cmd_key == "goodbye":
                interruptible_speak("À bientôt !")
                socketio.emit('status', {'message': 'JARVIS arrêté'})
                break
            elif cmd_key == "take_notes":
                prendre_des_notes()
            elif cmd_key == "add_tool":
                ajouter_outil_vocal()
            else:
                # Au cas où on ajoute d'autres commandes plus tard
                interruptible_speak("Commande reconnue, mais aucune action associée.")
        else:
            # Pas de commande claire : on envoie au modèle
            response = send_to_ollama(transcription, history=history)
            print("\n🤖 Réponse de Ollama :")
            print(response)
            interruptible_speak(response)
            history.append((transcription, response))

# --- Démarrage ---
if __name__ == "__main__":
    jarvis_thread = threading.Thread(target=run_jarvis)
    jarvis_thread.start()
    socketio.run(app, host='0.0.0.0', port=5000)
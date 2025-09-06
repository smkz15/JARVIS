# --- IMPORTS ---
import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import tempfile
import os
import queue
import time
import subprocess
import platform
import psutil
import shutil
import threading
import webbrowser
import re
import unicodedata
import importlib.util
from langdetect import detect
from ollama import chat

# --- PARAMÈTRES ---
SAMPLE_RATE = 16000
CHANNELS = 1
TRIGGER_WORD = "hello"
SILENCE_THRESHOLD_DB = -35
SILENCE_DURATION = 1
FRAME_DURATION = 0.5
OLLAMA_MODEL = "phi4-mini"
CODESRAL_MODEL = "codestral:latest"
LUMEN_FILE = "lumen10.py"
q = queue.Queue()
say_process = None
history = []

# --- PERSONNALITÉ ---
PERSONALITY_PROMPT = """
Tu es Phi, une assistante vocale intelligente, bienveillante et concise. 
Tu aides l'utilisateur à créer des outils et à répondre à ses questions de manière simple et claire.
"""

# --- IMPORT DYNAMIQUE ---
def import_lumen_functions():
    if not os.path.exists(LUMEN_FILE):
        return
    spec = importlib.util.spec_from_file_location("lumen10", LUMEN_FILE)
    lumen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lumen)
    globals().update({k: getattr(lumen, k) for k in dir(lumen) if callable(getattr(lumen, k))})

# --- LANCEMENT DES MODÈLES ---
def start_model_if_needed(model_name):
    for proc in psutil.process_iter(['name', 'cmdline']):
        try:
            if 'ollama' in proc.info['name'].lower() and model_name in ' '.join(proc.info['cmdline']):
                return
        except: continue
    if platform.system() == "Darwin":
        path = shutil.which("ollama")
        if path:
            script = f'''tell application "Terminal" to do script "{path} run {model_name}"'''
            subprocess.run(["osascript", "-e", script])

# --- AUDIO ---
def rms_db(audio):
    rms = np.sqrt(np.mean(audio ** 2))
    return -100 if rms == 0 else 20 * np.log10(rms)

def save_audio(audio, sr):
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wavfile.write(f.name, sr, audio)
    return f.name

def transcribe_audio(path):
    model = whisper.load_model("base")
    result = model.transcribe(path)
    text = result["text"].lower()
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    return re.sub(r'[^\w\s]', '', text).strip()

def detect_trigger_word():
    audio = sd.rec(int(1 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    text = transcribe_audio(save_audio(audio, SAMPLE_RATE))
    return TRIGGER_WORD in text.lower()

def audio_callback(indata, frames, time_info, status):
    if status: print("⚠️", status)
    q.put(indata.copy())

def dynamic_record_stream():
    recorded = []
    silence = 0
    frame_size = int(FRAME_DURATION * SAMPLE_RATE)
    with sd.InputStream(callback=audio_callback, channels=CHANNELS, samplerate=SAMPLE_RATE, blocksize=frame_size):
        while True:
            try: frame = q.get(timeout=1)
            except queue.Empty: break
            frame = frame.flatten()
            recorded.append(frame)
            db = rms_db(frame)
            silence = silence + FRAME_DURATION if db < SILENCE_THRESHOLD_DB else 0
            if silence > SILENCE_DURATION: break
    if not recorded: return np.zeros(int(SAMPLE_RATE * 1), dtype=np.int16)
    return np.concatenate(recorded)

# --- SYNTHÈSE VOCALE ---
def interruptible_speak(text):
    global say_process
    try:
        voice = "Thomas" if detect(text) == "fr" else "Samantha"
        say_process = subprocess.Popen(["say", "-v", voice, text])
        def interrupt_listener():
            with sd.InputStream(channels=1, samplerate=SAMPLE_RATE) as stream:
                while True:
                    audio = stream.read(1024)[0].flatten()
                    if rms_db(audio) > SILENCE_THRESHOLD_DB + 10:
                        say_process.terminate()
                        break
                    if say_process.poll() is not None:
                        break
        t = threading.Thread(target=interrupt_listener)
        t.start()
        say_process.wait()
        t.join()
    except Exception as e:
        print(f"Erreur synthèse : {e}")

# --- OLLAMA ---
def send_to_ollama(prompt, model=OLLAMA_MODEL, history=None):
    messages = [{"role": "system", "content": PERSONALITY_PROMPT}]
    if history:
        for u, r in history:
            messages += [{"role": "user", "content": u}, {"role": "assistant", "content": r}]
    messages.append({"role": "user", "content": prompt})
    try:
        return chat(model=model, messages=messages).message.content
    except Exception as e:
        return f"Erreur connexion : {e}"

# --- CODESTRAL ---
def coder_fonction_avec_codestral(demande):
    start_model_if_needed(CODESRAL_MODEL)
    path = os.path.realpath(__file__)
    try:
        with open(path, "r", encoding="utf-8") as f:
            main_code = f.read()
    except Exception as e:
        return None
    system_msg = (
        "Tu es un assistant Python. À partir du script principal d’un assistant vocal, génère UNE SEULE fonction complète à insérer dans lumen10.py. "
        "Ta réponse commence par 'def', sans explication ni commentaire.\n\n"
        f"Code principal :\n{main_code}"
    )
    try:
        response = chat(model=CODESRAL_MODEL, messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": demande}
        ]).message.content.strip()
        if not response.startswith("def "):
            print("⚠️ Mauvais code :", response)
            return None
        return response
    except Exception as e:
        print("Erreur Codestral :", e)
        return None

def insert_code_in_script(nom_fonction, code_fonction):
    try:
        if not os.path.exists(LUMEN_FILE):
            with open(LUMEN_FILE, "w") as f:
                f.write("# --- Début outils dynamiques ---\n# --- Fin outils dynamiques ---\n")
        with open(LUMEN_FILE, "r", encoding="utf-8") as f:
            content = f.read()
        if nom_fonction in content:
            print(f"🔁 Fonction '{nom_fonction}' déjà présente.")
            return
        nouveau_code = f"\n# --- Fonction ajoutée dynamiquement ---\n{code_fonction}\n"
        content = re.sub(r"(# --- Début outils dynamiques ---\n)(.*?)(# --- Fin outils dynamiques ---)",
                         lambda m: m.group(1) + m.group(2) + nouveau_code + m.group(3),
                         content, flags=re.DOTALL)
        with open(LUMEN_FILE, "w", encoding="utf-8") as f:
            f.write(content)
        import_lumen_functions()
        print(f"✅ Fonction '{nom_fonction}' ajoutée.")
    except Exception as e:
        print("Erreur insertion :", e)

# --- AJOUT OUTIL VOCAL ---
def ajouter_outil_vocal():
    interruptible_speak("Quel outil veux-tu que je code ?")
    demande = transcribe_audio(save_audio(dynamic_record_stream(), SAMPLE_RATE))
    code = coder_fonction_avec_codestral(demande)
    if code:
        match = re.search(r"def (\w+)\(", code)
        if match:
            nom_fonction = match.group(1)
            insert_code_in_script(nom_fonction, code)
            try:
                exec(code, globals())
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

# --- OUTILS INTÉGRÉS ---
def prendre_capture_ecran():
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.expanduser(f"~/Desktop/screenshot_{ts}.png")
    try:
        subprocess.run(["screencapture", "-x", path], check=True)
        interruptible_speak("Capture enregistrée.")
    except:
        interruptible_speak("Échec de la capture.")

def prendre_des_notes():
    interruptible_speak("Je t’écoute. Que dois-je noter ?")
    note = transcribe_audio(save_audio(dynamic_record_stream(), SAMPLE_RATE))
    note = note.replace('"', '\\"')
    script = f'''
    tell application "Notes"
        activate
        tell account "iCloud"
            make new note at folder "Notes" with properties {{name:"Note dictée", body:"{note}"}}
        end tell
    end tell
    '''
    subprocess.run(["osascript", "-e", script])
    interruptible_speak("C’est noté.")

# --- BOUCLE PRINCIPALE ---
if __name__ == "__main__":
    start_model_if_needed(OLLAMA_MODEL)
    start_model_if_needed(CODESRAL_MODEL)
    import_lumen_functions()
    in_convo = False

    while True:
        if not in_convo:
            if detect_trigger_word():
                in_convo = True
                interruptible_speak("Bonjour, je t'écoute.")
            continue

        audio = dynamic_record_stream()
        text = transcribe_audio(save_audio(audio, SAMPLE_RATE))
        print("📝", text)

        if "au revoir" in text:
            interruptible_speak("À bientôt !")
            break
        elif "capture ecran" in text:
            prendre_capture_ecran()
        elif "prendre des notes" in text:
            prendre_des_notes()
        elif "recherche sur wikipedia" in text:
            terme = text.replace("recherche sur wikipedia", "").strip()
            webbrowser.open(f"https://fr.wikipedia.org/wiki/{terme.replace(' ', '_')}")
            interruptible_speak(f"Voici {terme} sur Wikipédia.")
        elif "recherche sur github" in text:
            terme = text.replace("recherche sur github", "").strip()
            webbrowser.open(f"https://github.com/search?q={terme.replace(' ', '+')}")
            interruptible_speak(f"Voici {terme} sur GitHub.")
        elif text.strip() == "mod code":
            ajouter_outil_vocal()
        elif text.startswith("ajoute un outil pour"):
            ajouter_outil_vocal()
        else:
            response = send_to_ollama(text, history=history)
            interruptible_speak(response)
            history.append((text, response))

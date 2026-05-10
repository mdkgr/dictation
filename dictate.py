"""
Greek Dictation — Μίλα και γράψε στον δρομέα

Ctrl+Shift+Space : εναλλαγή εγγραφής / μεταγραφής
Ctrl+Shift+Q     : έξοδος

Pipeline: Gemini 2.5 Flash (audio → text streaming, incremental paste).
"""

import os
import io
import sys
import wave
import time
import threading
import ctypes
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import sounddevice as sd
import pyperclip
from google import genai
from google.genai import types

# Win32 polling for hotkeys — rock-solid, no hook degradation
if sys.platform == "win32":
    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _user32.GetAsyncKeyState.restype = ctypes.c_short
    _VK_CONTROL = 0x11
    _VK_SHIFT = 0x10
    _VK_MENU = 0x12  # Alt
    _VK_SPACE = 0x20
    _VK_Q = 0x51
    _VK_V = 0x56
    _VK_BACK = 0x08
    _KEYEVENTF_KEYUP = 0x0002

    def _is_down(vk):
        return bool(_user32.GetAsyncKeyState(vk) & 0x8000)

    def send_ctrl_v():
        _user32.keybd_event(_VK_CONTROL, 0, 0, 0)
        _user32.keybd_event(_VK_V, 0, 0, 0)
        _user32.keybd_event(_VK_V, 0, _KEYEVENTF_KEYUP, 0)
        _user32.keybd_event(_VK_CONTROL, 0, _KEYEVENTF_KEYUP, 0)

    def send_backspace():
        _user32.keybd_event(_VK_CONTROL, 0, _KEYEVENTF_KEYUP, 0)
        _user32.keybd_event(_VK_SHIFT, 0, _KEYEVENTF_KEYUP, 0)
        _user32.keybd_event(_VK_BACK, 0, 0, 0)
        _user32.keybd_event(_VK_BACK, 0, _KEYEVENTF_KEYUP, 0)
else:
    import keyboard

    def _is_down(vk):
        return False

    def send_ctrl_v():
        keyboard.send("ctrl+v")

    def send_backspace():
        keyboard.send("backspace")

# Separate output stream for beeps (avoids conflict with recording input stream)
_beep_stream_lock = threading.Lock()
_beep_cache = {}


def beep(freq=600, duration_ms=120):
    """Play a tone through speakers without blocking."""
    def _play():
        try:
            key = (freq, duration_ms)
            tone = _beep_cache.get(key)
            if tone is None:
                sr = 44100
                t = np.linspace(0, duration_ms / 1000, int(sr * duration_ms / 1000), dtype=np.float32)
                tone = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
                _beep_cache[key] = tone
            with _beep_stream_lock:
                sd.play(tone, samplerate=44100, blocking=True)
        except Exception:
            pass
    threading.Thread(target=_play, daemon=True).start()


def set_console_title(title):
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetConsoleTitleW(title)


# ── Config ────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = "gemini-2.5-flash"
HOTKEY = "ctrl+shift+space"
HOTKEY_NOSPACE = "ctrl+alt+space"
QUIT_KEY = "ctrl+shift+q"
SAMPLE_RATE = 16000
MAX_RETRIES = 3
BACKUP_DIR = Path(__file__).parent / "backups"

PROMPT = (
    "Απομαγνητοφώνησε αυτή την ηχογράφηση στα ελληνικά. "
    "Πρέπει να είσαι 100% ακριβής. Ακολούθησε αυτά τα βήματα:\n"
    "1. Μετέγραψε ακριβώς τι ακούς.\n"
    "2. Διόρθωσε ορθογραφία, τονισμό και στίξη.\n"
    "3. Πρόσεξε ιδιαίτερα ομόηχες λέξεις (ει/οι/η/ι/υ, ο/ω, ε/αι).\n"
    "4. Διατήρησε τον προφορικό τόνο και το νόημα.\n"
    "Επίστρεψε ΜΟΝΟ το τελικό διορθωμένο κείμενο, τίποτα άλλο."
)


class RecordingIndicator:
    """System tray icon that turns red while recording."""

    def __init__(self):
        self._recording = False
        self._icon = None
        threading.Thread(target=self._run, daemon=True).start()

    @staticmethod
    def _make_icon(color):
        from PIL import Image, ImageDraw
        img = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.ellipse([8, 8, 56, 56], fill=color)
        return img

    def _run(self):
        import pystray
        self._icon = pystray.Icon(
            "dictation",
            icon=self._make_icon('#888888'),
            title="Greek Dictation — Idle",
        )
        self._icon.run()

    def show(self):
        self._recording = True
        if self._icon:
            self._icon.icon = self._make_icon('#e53935')
            self._icon.title = "Greek Dictation — Recording"

    def hide(self):
        self._recording = False
        if self._icon:
            self._icon.icon = self._make_icon('#888888')
            self._icon.title = "Greek Dictation — Idle"


class Dictation:
    def __init__(self):
        self.recording = False
        self.processing = False
        self.frames = []
        self.stream = None
        self.strip_leading_space = False
        self._indicator = RecordingIndicator()
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    # ── Toggle ────────────────────────────────────────────────────────
    def toggle(self, strip_leading=False):
        if self.processing:
            print("  ⏳ Μεταγραφή σε εξέλιξη, περίμενε...")
            return
        if not self.recording:
            self.strip_leading_space = strip_leading
            self._start()
        else:
            self._stop()

    # ── Start recording ───────────────────────────────────────────────
    def _start(self):
        self.frames = []
        self.recording = True

        def callback(indata, frames, time_info, status):
            if self.recording:
                self.frames.append(indata.copy())

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            callback=callback,
        )
        self.stream.start()
        set_console_title("● REC — Greek Dictation")
        self._indicator.show()
        beep(600, 120)
        print("  🎙  Εγγραφή... (πάτα ξανά για στοπ)")

    # ── Stop recording ────────────────────────────────────────────────
    def _stop(self):
        self.recording = False
        self._indicator.hide()
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        set_console_title("Greek Dictation")
        beep(400, 120)
        print("  ⏳ Μεταγραφή...")
        self.processing = True
        threading.Thread(target=self._transcribe, daemon=True).start()

    # ── Audio helpers ────────────────────────────────────────────────
    @staticmethod
    def _audio_to_wav(audio_data: np.ndarray) -> bytes:
        """Convert int16 numpy array to WAV bytes."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())
        return buf.getvalue()

    def _save_backup(self, wav_bytes: bytes) -> Path:
        """Save WAV to backup dir, return path."""
        BACKUP_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = BACKUP_DIR / f"dictation_{ts}.wav"
        path.write_bytes(wav_bytes)
        return path

    def _transcribe_audio(self, audio_part):
        """Transcribe + proofread in a single Gemini call (streaming)."""
        chunks = []
        for chunk in self.client.models.generate_content_stream(
            model=MODEL,
            contents=[PROMPT, audio_part],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        ):
            if chunk.text:
                chunks.append(chunk.text)
        return "".join(chunks).replace("\n", " ").replace("\r", "").strip()

    def _paste_text(self, text):
        """Paste text at cursor position, restore clipboard."""
        try:
            old_clip = pyperclip.paste()
        except Exception:
            old_clip = ""
        pyperclip.copy(text)
        time.sleep(0.02)
        send_ctrl_v()
        time.sleep(0.08)
        try:
            pyperclip.copy(old_clip)
        except Exception:
            pass

    # ── Transcribe, proofread & paste ───────────────────────────────
    def _transcribe(self):
        try:
            if not self.frames:
                print("  ⚠  Δεν καταγράφηκε ήχος")
                beep(200, 300)
                return

            audio = np.concatenate(self.frames)
            duration_s = len(audio) / SAMPLE_RATE
            print(f"  📏 Διάρκεια: {duration_s:.1f}s")

            full_wav = self._audio_to_wav(audio)
            audio_part = types.Part.from_bytes(
                data=full_wav, mime_type="audio/wav"
            )

            text = ""
            t0 = time.perf_counter()
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    text = self._transcribe_audio(audio_part)
                    if text:
                        break
                except Exception as e:
                    if attempt < MAX_RETRIES:
                        wait = 2 ** attempt
                        print(f"  ⚠  {e}, retry {attempt}/{MAX_RETRIES} σε {wait}s...")
                        time.sleep(wait)
                    else:
                        raise
            elapsed = time.perf_counter() - t0
            print(f"  ⚡ API: {elapsed:.1f}s")

            if not text:
                backup_path = self._save_backup(full_wav)
                print("  ❌ Κενή απάντηση μετά από retries")
                print(f"  💾 Backup: {backup_path}")
                beep(200, 300)
                return

            if self.strip_leading_space:
                text = text.lstrip()
            print(f"  ✅ {text}")
            self._paste_text(text)
            beep(800, 80)

        except Exception as e:
            try:
                if self.frames:
                    audio = np.concatenate(self.frames)
                    backup_path = self._save_backup(self._audio_to_wav(audio))
                    print(f"  💾 Backup: {backup_path}")
            except Exception:
                pass
            print(f"  ❌ Σφάλμα: {e}")
            beep(200, 500)
        finally:
            self.processing = False
            set_console_title("Greek Dictation")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    if not GEMINI_API_KEY:
        print("❌ Δεν βρέθηκε GEMINI_API_KEY!")
        print()
        print("Ρύθμισε το ως environment variable:")
        print('  setx GEMINI_API_KEY "your-key-here"')
        sys.exit(1)

    d = Dictation()

    def hotkey_poll_loop():
        last_main = last_nospace = last_quit = False
        while True:
            try:
                ctrl = _is_down(_VK_CONTROL)
                shift = _is_down(_VK_SHIFT)
                alt = _is_down(_VK_MENU)
                space = _is_down(_VK_SPACE)
                q_key = _is_down(_VK_Q)

                main_combo = ctrl and shift and space and not alt
                nospace_combo = ctrl and alt and space and not shift
                quit_combo = ctrl and shift and q_key

                if main_combo and not last_main:
                    send_backspace()
                    d.toggle()
                if nospace_combo and not last_nospace:
                    d.toggle(strip_leading=True)
                if quit_combo and not last_quit:
                    os._exit(0)

                last_main = main_combo
                last_nospace = nospace_combo
                last_quit = quit_combo
            except Exception:
                pass
            time.sleep(0.04)

    threading.Thread(target=hotkey_poll_loop, daemon=True).start()

    set_console_title("Greek Dictation")
    print()
    print("╔════════════════════════════════════════════════╗")
    print("║  🎙  Greek Dictation                           ║")
    print("╠════════════════════════════════════════════════╣")
    print(f"║  {HOTKEY:20s}  εγγραφή/στοπ           ║")
    print(f"║  {HOTKEY_NOSPACE:20s}  εγγραφή (χωρίς space) ║")
    print(f"║  {QUIT_KEY:20s}  έξοδος                 ║")
    print("║  Gemini 2.5 Flash (audio, streaming)          ║")
    print("╚════════════════════════════════════════════════╝")
    print()

    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("\n👋 Τέλος!")


if __name__ == "__main__":
    main()

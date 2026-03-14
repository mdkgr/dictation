"""
Greek Dictation — Μίλα και γράψε στον δρομέα

Ctrl+Shift+Space : εναλλαγή εγγραφής / μεταγραφής
Ctrl+Shift+Q     : έξοδος

Χρησιμοποιεί Gemini Flash API για ελληνική μεταγραφή.
"""

import os
import io
import sys
import wave
import time
import threading

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import sounddevice as sd
import pyperclip
import keyboard
from google import genai
from google.genai import types

# Separate output stream for beeps (avoids conflict with recording input stream)
_beep_stream_lock = threading.Lock()


def beep(freq=600, duration_ms=120):
    """Play a tone through speakers without blocking."""
    def _play():
        try:
            sr = 44100
            t = np.linspace(0, duration_ms / 1000, int(sr * duration_ms / 1000), dtype=np.float32)
            tone = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
            with _beep_stream_lock:
                sd.play(tone, samplerate=sr, blocking=True)
        except Exception:
            pass
    threading.Thread(target=_play, daemon=True).start()


# ── Config ────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = "gemini-2.5-flash"
HOTKEY = "ctrl+shift+space"
QUIT_KEY = "ctrl+shift+q"
SAMPLE_RATE = 16000

PROMPT = (
    "Απομαγνητοφώνησε αυτή την ηχογράφηση στα ελληνικά. "
    "Διόρθωσε ορθογραφία, τόνους και στίξη. "
    "Διατήρησε τον προφορικό τόνο. "
    "Επίστρεψε ΜΟΝΟ το κείμενο, τίποτα άλλο."
)


class Dictation:
    def __init__(self):
        self.recording = False
        self.processing = False
        self.frames = []
        self.stream = None
        self.client = genai.Client(api_key=API_KEY)

    # ── Toggle ────────────────────────────────────────────────────────
    def toggle(self):
        if self.processing:
            print("  ⏳ Μεταγραφή σε εξέλιξη, περίμενε...")
            return
        if not self.recording:
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
        beep(600, 120)
        print("  🎙  Εγγραφή... (πάτα ξανά για στοπ)")

    # ── Stop recording ────────────────────────────────────────────────
    def _stop(self):
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        beep(400, 120)
        print("  ⏳ Μεταγραφή...")
        self.processing = True
        threading.Thread(target=self._transcribe, daemon=True).start()

    # ── Transcribe & paste ────────────────────────────────────────────
    def _transcribe(self):
        try:
            if not self.frames:
                print("  ⚠  Δεν καταγράφηκε ήχος")
                beep(200, 300)
                return

            # Build WAV in memory
            audio = np.concatenate(self.frames)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio.tobytes())

            audio_part = types.Part.from_bytes(
                data=buf.getvalue(), mime_type="audio/wav"
            )

            response = self.client.models.generate_content(
                model=MODEL,
                contents=[PROMPT, audio_part],
            )

            text = response.text.strip()
            if not text:
                print("  ⚠  Κενή απάντηση από Gemini")
                beep(200, 300)
                return

            print(f"  ✅ {text}")

            # Paste at cursor via clipboard
            try:
                old_clip = pyperclip.paste()
            except Exception:
                old_clip = ""

            pyperclip.copy(text)
            time.sleep(0.05)
            keyboard.send("ctrl+v")

            # Restore previous clipboard after paste completes
            time.sleep(0.3)
            try:
                pyperclip.copy(old_clip)
            except Exception:
                pass

            beep(800, 80)  # success

        except Exception as e:
            print(f"  ❌ Σφάλμα: {e}")
            beep(200, 500)
        finally:
            self.processing = False


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    if not API_KEY:
        print("❌ Δεν βρέθηκε GEMINI_API_KEY!")
        print()
        print("Ρύθμισε το ως environment variable:")
        print('  set GEMINI_API_KEY=your-key-here')
        print()
        print("Ή μόνιμα στα Windows:")
        print('  setx GEMINI_API_KEY "your-key-here"')
        sys.exit(1)

    d = Dictation()
    keyboard.add_hotkey(HOTKEY, d.toggle, suppress=True)
    keyboard.add_hotkey(QUIT_KEY, lambda: os._exit(0), suppress=True)

    print()
    print("╔══════════════════════════════════════════╗")
    print("║  🎙  Greek Dictation                     ║")
    print("╠══════════════════════════════════════════╣")
    print(f"║  {HOTKEY:20s}  εγγραφή/στοπ     ║")
    print(f"║  {QUIT_KEY:20s}  έξοδος           ║")
    print("║  Model: gemini-2.5-flash                ║")
    print("╚══════════════════════════════════════════╝")
    print()

    try:
        keyboard.wait()
    except KeyboardInterrupt:
        print("\n👋 Τέλος!")


if __name__ == "__main__":
    main()

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
    "Διόρθωσε ορθογραφία, τόνους και στίξη. "
    "Διατήρησε τον προφορικό τόνο. "
    "Επίστρεψε ΜΟΝΟ το κείμενο, τίποτα άλλο."
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

    def _stream_and_paste(self, audio_part):
        """Stream Gemini response and paste chunks as they arrive.

        Returns (text, pasted_anything, error).
        Caller decides retry policy: safe to retry only if not pasted_anything.
        """
        full_text = []
        pasted_anything = False
        error = None
        first_meaningful_chunk = True

        try:
            old_clip = pyperclip.paste()
        except Exception:
            old_clip = ""

        try:
            stream = self.client.models.generate_content_stream(
                model=MODEL,
                contents=[PROMPT, audio_part],
            )
            for resp_chunk in stream:
                text_chunk = resp_chunk.text or ""
                if not text_chunk:
                    continue

                # One paragraph: collapse newlines to spaces
                text_chunk = text_chunk.replace("\n", " ").replace("\r", "")

                # Strip leading whitespace on first meaningful chunk if requested
                if first_meaningful_chunk:
                    if self.strip_leading_space:
                        text_chunk = text_chunk.lstrip()
                    if not text_chunk:
                        continue
                    first_meaningful_chunk = False

                full_text.append(text_chunk)

                pyperclip.copy(text_chunk)
                time.sleep(0.02)
                keyboard.send("ctrl+v")
                time.sleep(0.04)
                pasted_anything = True
        except Exception as e:
            error = e
        finally:
            # Wait for last paste to settle, then restore clipboard
            if pasted_anything:
                time.sleep(0.08)
            try:
                pyperclip.copy(old_clip)
            except Exception:
                pass

        return "".join(full_text), pasted_anything, error

    # ── Transcribe & paste ────────────────────────────────────────────
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
            for attempt in range(1, MAX_RETRIES + 1):
                text, pasted, error = self._stream_and_paste(audio_part)

                if text and not error:
                    break  # Success

                if pasted:
                    # Partial paste already on cursor — cannot retry without dup
                    if error:
                        print(f"  ⚠  Διακοπή stream μετά partial paste ({error})")
                    break

                # Nothing pasted yet — safe to retry
                if attempt < MAX_RETRIES:
                    wait = 2 ** attempt
                    reason = error if error else "κενή απάντηση"
                    print(f"  ⚠  {reason}, retry {attempt}/{MAX_RETRIES} σε {wait}s...")
                    time.sleep(wait)
                elif error:
                    raise error

            if not text or not text.strip():
                backup_path = self._save_backup(full_wav)
                print("  ❌ Κενή απάντηση μετά από retries")
                print(f"  💾 Backup: {backup_path}")
                beep(200, 300)
                return

            print(f"  ✅ {text}")
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

    def register_hotkeys():
        keyboard.add_hotkey(HOTKEY, d.toggle)
        keyboard.add_hotkey(HOTKEY_NOSPACE, lambda: d.toggle(strip_leading=True))
        keyboard.add_hotkey(QUIT_KEY, lambda: os._exit(0))

    register_hotkeys()

    # Watchdog: re-register hotkeys periodically to prevent Windows
    # from silently dropping the low-level keyboard hook.
    def hotkey_watchdog():
        while True:
            time.sleep(30)
            if not d.recording and not d.processing:
                try:
                    keyboard.unhook_all()
                    register_hotkeys()
                except Exception:
                    pass

    threading.Thread(target=hotkey_watchdog, daemon=True).start()

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
        keyboard.wait()
    except KeyboardInterrupt:
        print("\n👋 Τέλος!")


if __name__ == "__main__":
    main()

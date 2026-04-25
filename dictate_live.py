"""
Greek Dictation (Live) — Real-time streaming μεταγραφή με Gemini Live API.

Ctrl+Shift+Space : εναλλαγή εγγραφής / στοπ
Ctrl+Shift+Q     : έξοδος

Pipeline: Gemini Flash Live (audio in → partial transcripts σε real-time WebSocket).
Partial transcripts γίνονται paste καθώς φτάνουν, ΕΝΩ μιλάς.

Παράλληλα με το dictate.py (single-shot baseline). Ίδιο GEMINI_API_KEY.
"""

import os
import sys
import time
import asyncio
import threading
import ctypes

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


if sys.platform == "win32":
    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _VK_CONTROL = 0x11
    _VK_V = 0x56
    _KEYEVENTF_KEYUP = 0x0002

    def send_ctrl_v():
        """Inject Ctrl+V via Win32 directly. Bypassing keyboard.send avoids
        degrading the keyboard library's own low-level hook (which is what
        listens for Ctrl+Shift+Space) under heavy paste volume."""
        _user32.keybd_event(_VK_CONTROL, 0, 0, 0)
        _user32.keybd_event(_VK_V, 0, 0, 0)
        _user32.keybd_event(_VK_V, 0, _KEYEVENTF_KEYUP, 0)
        _user32.keybd_event(_VK_CONTROL, 0, _KEYEVENTF_KEYUP, 0)
else:
    def send_ctrl_v():
        keyboard.send("ctrl+v")


# ── Config ────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
# Override via env if model id changes (preview names rotate)
MODEL = os.environ.get("GEMINI_LIVE_MODEL", "gemini-3.1-flash-live-preview")
HOTKEY = "ctrl+shift+space"
HOTKEY_NOSPACE = "ctrl+alt+space"
QUIT_KEY = "ctrl+shift+q"
SAMPLE_RATE = 16000
BLOCK_SIZE = 1600  # 100ms chunks @ 16kHz
RECEIVER_DRAIN_TIMEOUT = 5.0  # seconds to wait for final transcripts after stop
STOP_POLL_INTERVAL = 0.05      # how often async loop checks self.recording

# NOTE: response_modalities=["TEXT"] triggers WebSocket 1011 internal error on
# gemini-3.1-flash-live-preview (python-genai issue #2238). Use ["AUDIO"] and
# ignore the audio response — input_transcription is what we actually consume.
#
# Server VAD fires turn_complete on short silence (~1s) and won't be tamed by
# silence_duration_ms on this preview model — so disable it entirely and drive
# turn boundaries manually via activity_start/activity_end + audio_stream_end.
# This keeps a single turn open for the whole recording, regardless of pauses.
LIVE_CONFIG = {
    "response_modalities": ["AUDIO"],
    "input_audio_transcription": {},
    "realtime_input_config": {
        "automatic_activity_detection": {"disabled": True},
    },
}


class RecordingIndicator:
    """System tray icon that turns red while recording."""

    def __init__(self):
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
            "dictation_live",
            icon=self._make_icon('#888888'),
            title="Greek Dictation Live — Idle",
        )
        self._icon.run()

    def show(self):
        if self._icon:
            self._icon.icon = self._make_icon('#e53935')
            self._icon.title = "Greek Dictation Live — Recording"

    def hide(self):
        if self._icon:
            self._icon.icon = self._make_icon('#888888')
            self._icon.title = "Greek Dictation Live — Idle"


class AsyncBridge:
    """Run an asyncio event loop in a daemon thread, expose schedule() to sync code."""

    def __init__(self):
        self.loop = None
        ready = threading.Event()

        def run():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            ready.set()
            self.loop.run_forever()

        threading.Thread(target=run, daemon=True).start()
        ready.wait()

    def schedule(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop)


class Dictation:
    def __init__(self, refresh_hotkeys=None):
        self.recording = False
        self.processing = False
        self.strip_leading_space = False
        self._first_chunk = True
        self._old_clip = ""
        self._collected = []
        self._refresh_hotkeys = refresh_hotkeys

        self._indicator = RecordingIndicator()
        self.bridge = AsyncBridge()
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    # ── Hotkey entry ─────────────────────────────────────────────────
    def toggle(self, strip_leading=False):
        # Recording → stop (always honored, even if cleanup is queued)
        if self.recording:
            self._stop()
            return
        # Cleanup phase after stop → block until async finalizes
        if self.processing:
            print("  ⏳ Επεξεργασία σε εξέλιξη, περίμενε...")
            return
        self._start(strip_leading)

    def _start(self, strip_leading):
        self.recording = True
        self.strip_leading_space = strip_leading
        self._first_chunk = True
        self._collected = []
        try:
            self._old_clip = pyperclip.paste()
        except Exception:
            self._old_clip = ""

        set_console_title("● REC — Greek Dictation Live")
        self._indicator.show()
        beep(600, 120)
        print("  🎙  Εγγραφή... (μίλα — paste γίνεται live)")
        print("  ", end="", flush=True)

        self.bridge.schedule(self._run_session())

    def _stop(self):
        self.recording = False
        self.processing = True  # gate re-start until async finalizes
        self._indicator.hide()
        beep(400, 120)

    # ── Async session ────────────────────────────────────────────────
    async def _run_session(self):
        loop = asyncio.get_running_loop()
        audio_queue: asyncio.Queue = asyncio.Queue()
        sd_stream = None

        def audio_callback(indata, frames, time_info, status):
            if not self.recording:
                return
            try:
                pcm = indata.tobytes()
                loop.call_soon_threadsafe(audio_queue.put_nowait, pcm)
            except Exception:
                pass

        try:
            sd_stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=BLOCK_SIZE,
                callback=audio_callback,
            )
            sd_stream.start()

            async with self.client.aio.live.connect(
                model=MODEL, config=LIVE_CONFIG
            ) as session:
                # VAD is disabled in config — manually open the activity
                # window so the server starts transcribing audio.
                try:
                    await session.send_realtime_input(activity_start=types.ActivityStart())
                except Exception as e:
                    print(f"\n  ⚠  activity_start error: {e}")

                sender = asyncio.create_task(self._audio_sender(session, audio_queue))
                receiver = asyncio.create_task(self._transcript_receiver(session))

                # Poll for stop signal from sync thread
                while self.recording:
                    await asyncio.sleep(STOP_POLL_INTERVAL)

                # Sentinel to stop sender
                audio_queue.put_nowait(None)
                await sender

                # Close the activity, then close the audio stream. The server
                # finalizes transcription and emits turn_complete; without
                # these the receiver would hang waiting for more audio.
                try:
                    await session.send_realtime_input(activity_end=types.ActivityEnd())
                except Exception as e:
                    print(f"\n  ⚠  activity_end error: {e}")
                try:
                    await session.send_realtime_input(audio_stream_end=True)
                except Exception as e:
                    print(f"\n  ⚠  audio_stream_end error: {e}")

                # Drain receiver up to N seconds for trailing transcripts
                # (it returns when turn_complete arrives)
                try:
                    await asyncio.wait_for(receiver, timeout=RECEIVER_DRAIN_TIMEOUT)
                except asyncio.TimeoutError:
                    receiver.cancel()
                    try:
                        await receiver
                    except (asyncio.CancelledError, Exception):
                        pass
        except Exception as e:
            print(f"\n  ❌ Σφάλμα Live session: {e}")
            beep(200, 500)
        finally:
            if sd_stream is not None:
                try:
                    sd_stream.stop()
                    sd_stream.close()
                except Exception:
                    pass

            # Restore clipboard after last paste settles
            try:
                time.sleep(0.1)
                pyperclip.copy(self._old_clip)
            except Exception:
                pass

            full_text = "".join(self._collected).strip()
            if full_text:
                print(f"\n  ✅ {full_text}")
                beep(800, 80)
            else:
                print("\n  ⚠  Κενό transcript")
                beep(200, 300)

            set_console_title("Greek Dictation Live")
            self.processing = False

            # Heavy keyboard.send("ctrl+v") usage during paste degrades the
            # keyboard library's low-level hook on Windows, so the next
            # Ctrl+Shift+Space stops firing. Refresh hooks before idle.
            if self._refresh_hotkeys is not None:
                try:
                    self._refresh_hotkeys()
                except Exception as e:
                    print(f"  ⚠  hotkey refresh error: {e}")

    async def _audio_sender(self, session, audio_queue: asyncio.Queue):
        while True:
            pcm = await audio_queue.get()
            if pcm is None:
                break
            try:
                await session.send_realtime_input(
                    audio=types.Blob(
                        data=pcm,
                        mime_type=f"audio/pcm;rate={SAMPLE_RATE}",
                    )
                )
            except Exception as e:
                print(f"\n  ⚠  Audio send error: {e}")
                break

    async def _transcript_receiver(self, session):
        loop = asyncio.get_running_loop()
        async for msg in session.receive():
            content = getattr(msg, "server_content", None)
            if not content:
                continue

            transcript = getattr(content, "input_transcription", None)
            if transcript:
                text = getattr(transcript, "text", None)
                if text:
                    text = text.replace("\n", " ").replace("\r", "")
                    if self._first_chunk:
                        if self.strip_leading_space:
                            text = text.lstrip()
                        if text:
                            self._first_chunk = False
                            self._collected.append(text)
                            print(text, end="", flush=True)
                            await loop.run_in_executor(None, self._paste_chunk, text)
                    else:
                        self._collected.append(text)
                        print(text, end="", flush=True)
                        await loop.run_in_executor(None, self._paste_chunk, text)

            # Exit only if user has stopped — guards against unexpected
            # turn_complete (e.g. if VAD config is ignored by a future API
            # version) so the receiver doesn't die mid-recording.
            if getattr(content, "turn_complete", False) and not self.recording:
                return

    @staticmethod
    def _paste_chunk(text):
        pyperclip.copy(text)
        time.sleep(0.02)
        send_ctrl_v()
        time.sleep(0.04)


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    if not GEMINI_API_KEY:
        print("❌ Δεν βρέθηκε GEMINI_API_KEY!")
        print()
        print("Ρύθμισε το ως environment variable:")
        print('  setx GEMINI_API_KEY "your-key-here"')
        sys.exit(1)

    def register_hotkeys():
        keyboard.add_hotkey(HOTKEY, d.toggle)
        keyboard.add_hotkey(HOTKEY_NOSPACE, lambda: d.toggle(strip_leading=True))
        keyboard.add_hotkey(QUIT_KEY, lambda: os._exit(0))

    def refresh_hotkeys():
        keyboard.unhook_all()
        register_hotkeys()

    d = Dictation(refresh_hotkeys=refresh_hotkeys)
    register_hotkeys()

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

    set_console_title("Greek Dictation Live")
    print()
    print("╔════════════════════════════════════════════════╗")
    print("║  🎙  Greek Dictation (Live)                    ║")
    print("╠════════════════════════════════════════════════╣")
    print(f"║  {HOTKEY:20s}  εγγραφή/στοπ           ║")
    print(f"║  {HOTKEY_NOSPACE:20s}  εγγραφή (χωρίς space) ║")
    print(f"║  {QUIT_KEY:20s}  έξοδος                 ║")
    print(f"║  {MODEL:44s}  ║")
    print("╚════════════════════════════════════════════════╝")
    print()

    try:
        keyboard.wait()
    except KeyboardInterrupt:
        print("\n👋 Τέλος!")


if __name__ == "__main__":
    main()

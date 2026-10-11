from __future__ import annotations

import logging
import queue
import shutil
import subprocess
import threading


class Speaker:
    """Best-effort offline speech via the OS default audio sink (Bluetooth or wired)."""
    def __init__(self, enabled: bool, language: str):
        self.language = language
        self.binary = (shutil.which("espeak-ng") or shutil.which("espeak")) if enabled else None
        self.queue: queue.Queue[str | None] = queue.Queue(maxsize=2)
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def say(self, person: str) -> None:
        if self.binary:
            try:
                self.queue.put_nowait(f"Xin chào {person}")
            except queue.Full:
                pass

    def _worker(self) -> None:
        while True:
            phrase = self.queue.get()
            if phrase is None:
                return
            try:
                subprocess.run([self.binary, "-v", self.language, phrase], timeout=12,
                               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                logging.getLogger(__name__).warning("Audio unavailable; recognition continues")

    def close(self) -> None:
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass

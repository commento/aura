from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Queue
from threading import Lock, Thread
import wave

import numpy as np

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - optional on dev machines
    sd = None


@dataclass
class AudioFeatures:
    rms: float = 0.0
    peak: float = 0.0
    spectral_centroid: float = 0.0


class AudioAnalyzer:
    def __init__(self, sample_rate: int, block_size: int, device: str | int | None = None):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.device = device
        self._features = AudioFeatures()
        self._lock = Lock()
        self._stream = None
        self._wave_file: wave.Wave_write | None = None
        self._record_queue: Queue[bytes] | None = None
        self._record_thread: Thread | None = None
        self._recorded_frames = 0
        self._stream_started = False

    def start(self) -> None:
        if sd is None:
            return

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()
        self._stream_started = True

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._record_thread is not None:
            assert self._record_queue is not None
            self._record_queue.put(b"")
            self._record_thread.join(timeout=2)
            self._record_thread = None
            self._record_queue = None
        if self._wave_file is not None:
            self._wave_file.close()
            self._wave_file = None

    def start_recording(self, path: str) -> None:
        if self._wave_file is not None:
            self._wave_file.close()
        self._wave_file = wave.open(path, "wb")
        self._wave_file.setnchannels(1)
        self._wave_file.setsampwidth(2)
        self._wave_file.setframerate(self.sample_rate)
        self._recorded_frames = 0
        self._record_queue = Queue(maxsize=64)
        self._record_thread = Thread(target=self._record_worker, daemon=True)
        self._record_thread.start()

    def read(self) -> AudioFeatures:
        with self._lock:
            return AudioFeatures(**self._features.__dict__)

    def _callback(self, indata, frames, time_info, status) -> None:  # pragma: no cover - realtime callback
        samples = np.squeeze(indata.astype(np.float32))
        if samples.size == 0:
            return

        rms = float(np.sqrt(np.mean(np.square(samples))))
        peak = float(np.max(np.abs(samples)))

        spectrum = np.abs(np.fft.rfft(samples))
        freqs = np.fft.rfftfreq(samples.size, d=1.0 / self.sample_rate)
        spectral_sum = float(np.sum(spectrum))
        centroid = float(np.sum(freqs * spectrum) / spectral_sum) if spectral_sum > 0 else 0.0

        with self._lock:
            self._features = AudioFeatures(rms=rms, peak=peak, spectral_centroid=centroid)

        if self._wave_file is not None:
            pcm16 = np.clip(samples, -1.0, 1.0)
            pcm16 = (pcm16 * 32767.0).astype(np.int16)
            if self._record_queue is not None:
                try:
                    self._record_queue.put_nowait(pcm16.tobytes())
                    self._recorded_frames += int(samples.size)
                except Exception:
                    pass

    def _record_worker(self) -> None:
        if self._record_queue is None:
            return
        while True:
            try:
                chunk = self._record_queue.get(timeout=0.2)
            except Empty:
                continue
            if chunk == b"":
                break
            if self._wave_file is not None:
                self._wave_file.writeframes(chunk)

    @property
    def recorded_duration(self) -> float:
        return self._recorded_frames / max(float(self.sample_rate), 1.0)

    @property
    def stream_started(self) -> bool:
        return self._stream_started

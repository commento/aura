from __future__ import annotations

import subprocess
import threading
import time
from subprocess import TimeoutExpired
from datetime import datetime
from pathlib import Path

import numpy as np


class FfmpegRecorder:
    def __init__(
        self,
        ffmpeg_bin: str,
        output_path: str,
        width: int,
        height: int,
        fps: int,
        video_codec: str,
        pixel_format: str,
        crf: int,
        preset: str,
        audio_enabled: bool,
        audio_input_format: str,
        audio_input_device: str,
    ):
        timestamped = datetime.now().strftime(output_path)
        self.output_file = Path(timestamped)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.video_only_file = self.output_file.with_suffix(".video.mp4")
        self.width = width
        self.height = height
        self.fps = fps
        self._frame_interval = 1.0 / max(1, fps)
        self._latest_frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self.process = self._start_process(
            ffmpeg_bin=ffmpeg_bin,
            output=str(self.video_only_file),
            video_codec=video_codec,
            pixel_format=pixel_format,
            crf=crf,
            preset=preset,
            audio_enabled=False,
            audio_input_format=audio_input_format,
            audio_input_device=audio_input_device,
        )
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()
        self.ffmpeg_bin = ffmpeg_bin

    def _start_process(
        self,
        ffmpeg_bin: str,
        output: str,
        video_codec: str,
        pixel_format: str,
        crf: int,
        preset: str,
        audio_enabled: bool,
        audio_input_format: str,
        audio_input_device: str,
    ) -> subprocess.Popen:
        command = [
            ffmpeg_bin,
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(self.fps),
            "-i",
            "-",
        ]

        command.extend(
            [
                "-c:v",
                video_codec,
                "-preset",
                preset,
                "-crf",
                str(crf),
                "-pix_fmt",
                pixel_format,
                "-shortest",
            ]
        )

        command.extend(["-an"])

        command.append(output)
        return subprocess.Popen(command, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        with self._lock:
            self._latest_frame = frame.copy()

    def _writer_loop(self) -> None:
        next_frame_time = time.monotonic()
        while not self._stop_event.is_set():
            with self._lock:
                frame = None if self._latest_frame is None else self._latest_frame.copy()

            if frame is not None and self.process.stdin is not None:
                try:
                    self.process.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    self._stop_event.set()
                    break

            next_frame_time += self._frame_interval
            sleep_for = next_frame_time - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_frame_time = time.monotonic()

    def close(self) -> None:
        self._stop_event.set()
        if hasattr(self, "_writer_thread"):
            self._writer_thread.join(timeout=2)
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=3)
        except TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)

    def mux_audio(self, audio_path: str) -> None:
        audio_file = Path(audio_path)
        if not audio_file.exists() or audio_file.stat().st_size == 0:
            self.video_only_file.replace(self.output_file)
            return

        muxed_tmp = self.output_file.with_suffix(".mux.mp4")
        command = [
            self.ffmpeg_bin,
            "-y",
            "-i",
            str(self.video_only_file),
            "-i",
            str(audio_file),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(muxed_tmp),
        ]
        subprocess.run(command, check=True)
        muxed_tmp.replace(self.output_file)
        if self.video_only_file.exists():
            self.video_only_file.unlink()


#!/usr/bin/env python3
"""
MacBook demo: webcam + live audio -> progressively added fisheye distortion.

Requirements:
    pip install opencv-python sounddevice numpy

Run:
    python mys_audio_fisheye_demo.py

Optional flags:
    python mys_audio_fisheye_demo.py --camera 0 --width 1280 --height 720
"""

import argparse
import queue
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
import sounddevice as sd


@dataclass
class AudioState:
    instant_rms: float = 0.0
    smooth_rms: float = 0.0
    peak_hold: float = 0.0
    last_update: float = 0.0


class AudioMonitor:
    def __init__(self, samplerate: int = 48000, blocksize: int = 1024, device=None):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.device = device
        self.state = AudioState()
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=32)
        self._stop = threading.Event()
        self._thread = None
        self._stream = None

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            # Non-fatal: dropouts can happen.
            pass
        mono = np.mean(indata, axis=1).astype(np.float32, copy=False)
        try:
            self._queue.put_nowait(mono.copy())
        except queue.Full:
            # Drop oldest-ish behavior by ignoring overflow.
            pass

    def _worker(self):
        attack = 0.25
        release = 0.03
        peak_decay = 0.992

        while not self._stop.is_set():
            try:
                chunk = self._queue.get(timeout=0.1)
            except queue.Empty:
                self.state.smooth_rms *= 0.995
                self.state.peak_hold *= peak_decay
                continue

            rms = float(np.sqrt(np.mean(np.square(chunk)) + 1e-12))
            self.state.instant_rms = rms

            if rms > self.state.smooth_rms:
                self.state.smooth_rms = (1.0 - attack) * self.state.smooth_rms + attack * rms
            else:
                self.state.smooth_rms = (1.0 - release) * self.state.smooth_rms + release * rms

            self.state.peak_hold = max(self.state.peak_hold * peak_decay, self.state.smooth_rms)
            self.state.last_update = time.time()

    def start(self):
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            channels=2,
            dtype="float32",
            device=0,
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self):
        self._stop.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def smoothstep(edge0, edge1, x):
    if edge0 == edge1:
        return 0.0
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def build_fisheye_maps(width: int, height: int, strength: float):
    """
    strength = 0.0 => identity
    strength ~ 0.7 => noticeable fisheye / barrel distortion
    """
    y, x = np.indices((height, width), dtype=np.float32)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    xn = (x - cx) / max(width, 1)
    yn = (y - cy) / max(height, 1)
    r = np.sqrt(xn * xn + yn * yn)

    # Added fisheye: push content outward with a radial polynomial.
    # For each output pixel, sample from a less distorted inner radius.
    k = float(strength)
    src_r = r / (1.0 + k * (r ** 2) * 4.0 + 1e-9)

    scale = np.ones_like(r, dtype=np.float32)
    mask = r > 1e-6
    scale[mask] = src_r[mask] / r[mask]

    map_x = cx + (x - cx) * scale
    map_y = cy + (y - cy) * scale

    return map_x.astype(np.float32), map_y.astype(np.float32)


def add_minimal_white_aura(frame, energy):
    """
    Minimal white aura centered in the frame.
    This is a stylized prototype, not performer tracking yet.
    """
    h, w = frame.shape[:2]
    y, x = np.indices((h, w), dtype=np.float32)

    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    # Soft vertically elongated field, reminiscent of a body-stage presence.
    xn = (x - cx) / max(w * 0.22, 1.0)
    yn = (y - cy) / max(h * 0.34, 1.0)
    r = np.sqrt(xn * xn + yn * yn)

    # Aura grows with energy but stays minimal.
    radius = 0.72 + 0.28 * energy
    edge_softness = 0.16 + 0.08 * energy

    aura = np.clip(1.0 - (r - radius + edge_softness) / max(edge_softness, 1e-6), 0.0, 1.0)
    aura = cv2.GaussianBlur(aura, (0, 0), sigmaX=10.0 + 18.0 * energy, sigmaY=14.0 + 24.0 * energy)

    # Hollow center to keep it airy/minimal instead of becoming a solid white blob.
    inner_xn = (x - cx) / max(w * (0.12 + 0.03 * energy), 1.0)
    inner_yn = (y - cy) / max(h * (0.22 + 0.04 * energy), 1.0)
    inner_r = np.sqrt(inner_xn * inner_xn + inner_yn * inner_yn)
    hollow = np.clip(1.0 - inner_r, 0.0, 1.0)
    hollow = cv2.GaussianBlur(hollow, (0, 0), sigmaX=8.0, sigmaY=10.0)

    aura = np.clip(aura - 0.55 * hollow, 0.0, 1.0)

    # Add a faint vertical spine to suggest a unified luminous presence.
    spine = np.exp(-(((x - cx) / max(w * 0.035, 1.0)) ** 2))
    spine *= np.exp(-(((y - cy) / max(h * 0.40, 1.0)) ** 2))
    aura = np.clip(aura + 0.22 * energy * spine, 0.0, 1.0)

    alpha = 0.05 + 0.32 * energy
    aura_rgb = np.dstack([aura, aura, aura]).astype(np.float32)

    out = frame.astype(np.float32) / 255.0
    out = np.clip(out + alpha * aura_rgb, 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def overlay_hud(frame, energy, raw_rms, gain, strength):
    h, w = frame.shape[:2]
    bar_w = int(w * 0.30)
    x0, y0 = 30, h - 50

    cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + 18), (40, 40, 40), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + int(bar_w * energy), y0 + 18), (255, 255, 255), -1)

    lines = [
        f"audio energy: {energy:.2f}",
        f"raw rms: {raw_rms:.4f}",
        f"gain: {gain:.1f}x",
        f"fisheye strength: {strength:.2f}",
        "keys: q quit | +/- gain | [ ] distortion max",
    ]

    y = 28
    for line in lines:
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--audio-device", default=None)
    parser.add_argument("--gain", type=float, default=18.0, help="audio sensitivity")
    parser.add_argument("--max-strength", type=float, default=0.72, help="maximum fisheye intensity")
    parser.add_argument("--mirror", action="store_true", help="mirror the webcam preview")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Could not open camera.")

    height, width = frame.shape[:2]

    audio = AudioMonitor(device=args.audio_device)
    audio.start()

    gain = float(args.gain)
    max_strength = float(args.max_strength)

    last_strength_bin = -1
    cached_maps = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if args.mirror:
                frame = cv2.flip(frame, 1)

            # Convert smoothed RMS into 0..1 energy.
            raw = audio.state.smooth_rms
            boosted = np.clip(raw * gain, 0.0, 1.0)
            energy = float(smoothstep(0.03, 0.55, boosted))

            # Gradually "add fisheye" as audio grows.
            strength = max_strength * energy

            # Cache maps in coarse bins to avoid rebuilding every frame.
            strength_bin = int(strength * 100)
            if strength_bin != last_strength_bin:
                cached_maps = build_fisheye_maps(width, height, strength)
                last_strength_bin = strength_bin

            map_x, map_y = cached_maps
            distorted = cv2.remap(
                frame,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT101,
            )

            # Optional subtle glow blend at high energy.
            if energy > 0.45:
                blur = cv2.GaussianBlur(distorted, (0, 0), sigmaX=8.0 * energy, sigmaY=8.0 * energy)
                alpha = 0.08 + 0.18 * energy
                distorted = cv2.addWeighted(distorted, 1.0 - alpha, blur, alpha, 0)

            # Minimal white aura driven by audio energy.
            distorted = add_minimal_white_aura(distorted, energy)

            overlay_hud(distorted, energy, raw, gain, strength)
            cv2.imshow("MYS audio fisheye demo", distorted)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key in (ord("+"), ord("=")):
                gain = min(60.0, gain + 1.0)
            elif key == ord("-"):
                gain = max(1.0, gain - 1.0)
            elif key == ord("]"):
                max_strength = min(1.2, max_strength + 0.05)
            elif key == ord("["):
                max_strength = max(0.05, max_strength - 0.05)

    finally:
        audio.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

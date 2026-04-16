"""
SSVEP demo + OpenBCI CCA pipeline (same signal path as guess.py) in one process.

Main thread: pygame flicker UI (16 Hz | 31 Hz, subdivisions, inactive regions grayed).
Background: Cyton stream runs `process_sample` (bandpass, CCA vs reference, baseline).

Run (hardware):
  OPENBCI_PORT=/dev/cu.usbserial-XXXX python ssvep_grid_test.py

No board (UI / keys only):
  python ssvep_grid_test.py --keyboard-only
"""

from __future__ import annotations

import argparse
import itertools
import queue
import sys
import threading
from typing import Final

import numpy as np
import pygame
from pyOpenBCI import OpenBCICyton
from scipy.signal import butter, filtfilt
from sklearn.cross_decomposition import CCA

# ---------------------------------------------------------------------------
# Tunable parameters — keep values here (no scattered literals).
# ---------------------------------------------------------------------------

# Serial (override with env OPENBCI_PORT or --port)
DEFAULT_OPENBCI_PORT: Final[str] = "/dev/ttyUSB0"
ENV_OPENBCI_PORT_KEY: Final[str] = "OPENBCI_PORT"

# Sampling & buffer (must match Cyton / acquisition settings)
SAMPLING_RATE_HZ: Final[int] = 250
CCA_WINDOW_SAMPLES: Final[int] = SAMPLING_RATE_HZ
NUM_OPENBCI_CHANNELS: Final[int] = 8

# Board column indices into sample.channels_data used as EEG features for CCA
# (same choice as guess.py: channels 1–6 of the 8-channel frame)
CHANNEL_INDICES_FOR_CCA: Final[tuple[int, ...]] = (1, 2, 3, 4, 5, 6)

# SSVEP targets used both for signal references and on-screen flicker
SSVEP_FREQUENCIES_HZ: Final[tuple[int, int]] = (16, 31)

# z-score-style threshold on (corr − baseline) per frequency (from guess.py)
CORR_DELTA_THRESHOLDS: Final[dict[int, float]] = {
    SSVEP_FREQUENCIES_HZ[0]: 0.11,
    SSVEP_FREQUENCIES_HZ[1]: 0.10,
}

# How many full CCA windows to average for resting baseline (guess.py)
CALIBRATION_FULL_WINDOWS: Final[int] = 500

# Log calibration percent to stderr every N windows while calibrating
CALIBRATION_LOG_INTERVAL_WINDOWS: Final[int] = 25

# Spinner stride in stream samples (visual/debug; not used for decisions)
SPINNER_UPDATE_STRIDE_SAMPLES: Final[int] = 30

# Harmonics k = 1 .. (exclusive upper bound) for sine/cos reference construction
REFERENCE_HARMONIC_K_START: Final[int] = 1
REFERENCE_HARMONIC_K_END_EXCLUSIVE: Final[int] = 3

# Bandpass on EEG snippets before CCA (Hz)
BANDPASS_LOW_HZ: Final[float] = 5.0
BANDPASS_HIGH_HZ: Final[float] = 40.0
BANDPASS_ORDER: Final[int] = 4

# CCA
CCA_N_COMPONENTS: Final[int] = 1

# How many *CCA detections* (same event that increments guess.py’s distribution)
# before the matching on-screen region subdivides
SUBDIVISION_HIT_THRESHOLD: Final[int] = 200

# Queue: if UI stalls, drop overflow detections rather than blocking acquisition
MAIN_THREAD_EVENT_QUEUE_MAXSIZE: Final[int] = 256

# --keyboard-only: wait this many UI frames before enabling fake hits (not EEG-based)
KEYBOARD_ONLY_CALIBRATION_FRAME_DIVISOR: Final[int] = 10
KEYBOARD_ONLY_DEFAULT_CALIBRATION_FRAMES: Final[int] = (
    max(1, CALIBRATION_FULL_WINDOWS // KEYBOARD_ONLY_CALIBRATION_FRAME_DIVISOR)
)

# pygame
TARGET_FPS: Final[int] = 60
WINDOW_WIDTH_PX: Final[int] = 1200
WINDOW_HEIGHT_PX: Final[int] = 800
MIN_WINDOW_DIMENSION_PX: Final[int] = 50
HUD_PRIMARY_FONT_SIZE_PX: Final[int] = 22
HUD_SECONDARY_FONT_SIZE_PX: Final[int] = 16
HUD_MARGIN_LEFT_PX: Final[int] = 12
HUD_LINE_STEP_PX: Final[int] = 22
HUD_BOTTOM_BLOCK_HEIGHT_PX: Final[int] = 52
BACKGROUND_GRAY: Final[tuple[int, int, int]] = (90, 90, 90)
PANEL_BORDER_GRAY: Final[tuple[int, int, int]] = (40, 40, 40)
INACTIVE_PANEL_GRAY: Final[tuple[int, int, int]] = (128, 128, 128)
FLASH_DARK: Final[tuple[int, int, int]] = (0, 0, 0)
FLASH_BRIGHT: Final[tuple[int, int, int]] = (255, 255, 255)
LABEL_TEXT_COLOR: Final[tuple[int, int, int]] = (180, 180, 255)
HUD_TEXT_COLOR: Final[tuple[int, int, int]] = (240, 240, 240)

# Flicker timing: black/white edge rate matches demo-style SSVEP drive
MS_PER_SECOND: Final[float] = 1000.0
FLASH_STATE_TOGGLES_PER_SECOND_FACTOR: Final[int] = 2  # two edges per fundamental cycle


# ---------------------------------------------------------------------------
# Derived / shared references
# ---------------------------------------------------------------------------

_SPINNER_FRAMES: Final[tuple[str, ...]] = tuple(
    [
        "⠁  ",
        "⠃  ",
        "⠇  ",
        "⡇  ",
        "⣇  ",
        "⣧  ",
        "⣷  ",
        "⣿  ",
        "⣾  ",
        "⣼  ",
        "⣸  ",
        "⢸  ",
        "⠸  ",
        "⠘  ",
        "⠈  ",
        "   ",
    ]
)


def _build_reference_templates() -> list[np.ndarray]:
    t = np.arange(CCA_WINDOW_SAMPLES, dtype=np.float64) / float(SAMPLING_RATE_HZ)
    blocks: list[np.ndarray] = []
    for f_hz in SSVEP_FREQUENCIES_HZ:
        cols: list[np.ndarray] = []
        for k in range(REFERENCE_HARMONIC_K_START, REFERENCE_HARMONIC_K_END_EXCLUSIVE):
            cols.append(np.sin(2 * np.pi * k * f_hz * t))
            cols.append(np.cos(2 * np.pi * k * f_hz * t))
        blocks.append(np.stack(cols, axis=1))
    return blocks


def bandpass_columns(data: np.ndarray, fs: int) -> np.ndarray:
    nyq = 0.5 * float(fs)
    band = (BANDPASS_LOW_HZ / nyq, BANDPASS_HIGH_HZ / nyq)
    b, a = butter(BANDPASS_ORDER, band, btype="band")
    return filtfilt(b, a, data, axis=0)


class EegPipeline:
    """Thread-safe state + process_sample callback (logic aligned with guess.py)."""

    def __init__(self, outbound: queue.Queue[tuple[str, object]]) -> None:
        self._lock = threading.Lock()
        self._outbound = outbound
        self.buffer = np.zeros((NUM_OPENBCI_CHANNELS, CCA_WINDOW_SAMPLES))
        self.samples_counted = 0
        self.is_calibrated = False
        self.static_baseline = np.zeros(len(SSVEP_FREQUENCIES_HZ), dtype=np.float64)
        self.all_calib_corrs: list[list[float]] = []
        self.Y_ref = _build_reference_templates()
        self.cca = CCA(n_components=CCA_N_COMPONENTS)
        self._spinner_iter = itertools.cycle(_SPINNER_FRAMES)
        self.spin_symbol = next(self._spinner_iter)
        self.detection_totals: dict[int, int] = {f: 0 for f in SSVEP_FREQUENCIES_HZ}

    def reset_calibration(self) -> None:
        with self._lock:
            self.buffer.fill(0)
            self.samples_counted = 0
            self.is_calibrated = False
            self.static_baseline.fill(0.0)
            self.all_calib_corrs.clear()
            self.detection_totals = {f: 0 for f in SSVEP_FREQUENCIES_HZ}

    def process_sample(self, sample: object) -> None:
        with self._lock:
            if self.samples_counted % SPINNER_UPDATE_STRIDE_SAMPLES == 0:
                self.spin_symbol = next(self._spinner_iter)

            self.buffer = np.roll(self.buffer, -1, axis=1)
            self.buffer[:, -1] = sample.channels_data

            idx = np.array(CHANNEL_INDICES_FOR_CCA, dtype=int)
            if not np.all(self.buffer[idx, -1] != 0):
                return

            x = self.buffer[idx].T
            x = bandpass_columns(x, SAMPLING_RATE_HZ)

            corrs: list[float] = []
            for y in self.Y_ref:
                self.cca.fit(x, y)
                x_c, y_c = self.cca.transform(x, y)
                corrs.append(float(np.corrcoef(x_c.T, y_c.T)[0, 1]))

            if not self.is_calibrated:
                self.all_calib_corrs.append(corrs)
                self.samples_counted += 1

                if self.samples_counted % CALIBRATION_LOG_INTERVAL_WINDOWS == 0:
                    pct = int((self.samples_counted / float(CALIBRATION_FULL_WINDOWS)) * 100.0)
                    print(f"Calibrating… {pct}%", file=sys.stderr, flush=True)

                if self.samples_counted >= CALIBRATION_FULL_WINDOWS:
                    self.static_baseline = np.mean(np.array(self.all_calib_corrs), axis=0)
                    self.is_calibrated = True
                    print("--- CALIBRATION COMPLETE ---", file=sys.stderr, flush=True)
                    print(f"Baseline correlation profile: {self.static_baseline}", file=sys.stderr, flush=True)
                    try:
                        self._outbound.put_nowait(("eeg_ready", tuple(float(x) for x in self.static_baseline)))
                    except queue.Full:
                        pass
                return

            rel = np.array(corrs, dtype=np.float64) - self.static_baseline

            for freq, score in zip(SSVEP_FREQUENCIES_HZ, rel):
                if score > CORR_DELTA_THRESHOLDS[freq]:
                    self.detection_totals[freq] += 1
                    msg = f"Detected: {freq} Hz {self.spin_symbol}"
                    sys.stdout.write("\r" + msg)
                    sys.stdout.flush()
                    try:
                        self._outbound.put_nowait(("detection", int(freq)))
                    except queue.Full:
                        pass
                    break

            self.samples_counted += 1


class FlashPanel:
    def __init__(self, rect: tuple[int, int, int, int], freq: int, depth: int = 0) -> None:
        self.rect = pygame.Rect(rect)
        self.freq = int(freq)
        self.depth = depth
        self.interval_ms = MS_PER_SECOND / (float(self.freq) * float(FLASH_STATE_TOGGLES_PER_SECOND_FACTOR))
        self.state = True
        self.last_toggle_ms = pygame.time.get_ticks()
        self.active = True

    def deactivate_tree(self) -> None:
        self.active = False

    def update(self, now_ms: int) -> None:
        if not self.active:
            return
        if now_ms - self.last_toggle_ms >= self.interval_ms:
            self.state = not self.state
            self.last_toggle_ms = now_ms

    def draw(self, surface: pygame.Surface, font: pygame.font.Font | None) -> None:
        if not self.active:
            color = INACTIVE_PANEL_GRAY
        else:
            color = FLASH_DARK if self.state else FLASH_BRIGHT
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, PANEL_BORDER_GRAY, self.rect, 1)
        if font is not None and self.active:
            label = font.render(f"{self.freq} Hz", True, LABEL_TEXT_COLOR)
            surface.blit(label, (self.rect.x + HUD_MARGIN_LEFT_PX // 2, self.rect.y + HUD_MARGIN_LEFT_PX // 2))


class FreqSplit:
    """Two children: SSVEP_FREQUENCIES_HZ[0] and [1], orientation alternates by depth."""

    def __init__(self, rect: tuple[int, int, int, int], depth: int = 0) -> None:
        self.rect = pygame.Rect(rect)
        self.depth = depth
        self.active = True
        bands = SSVEP_FREQUENCIES_HZ
        self._split_vertical = depth % 2 == 0
        x, y, w, h = self.rect
        child_depth = depth + 1
        if self._split_vertical:
            half = w // 2
            self.children: list[FlashPanel | FreqSplit] = [
                FlashPanel((x, y, half, h), bands[0], child_depth),
                FlashPanel((x + half, y, w - half, h), bands[1], child_depth),
            ]
        else:
            half = h // 2
            self.children = [
                FlashPanel((x, y, w, half), bands[0], child_depth),
                FlashPanel((x, y + half, w, h - half), bands[1], child_depth),
            ]

    def deactivate_tree(self) -> None:
        self.active = False
        for ch in self.children:
            ch.deactivate_tree()

    def update(self, now_ms: int) -> None:
        if not self.active:
            return
        for ch in self.children:
            ch.update(now_ms)

    def draw(self, surface: pygame.Surface, font: pygame.font.Font | None) -> None:
        for ch in self.children:
            ch.draw(surface, font)

    def subdivide_matching(self, freq: int) -> bool:
        if not self.active:
            return False
        for i, child in enumerate(self.children):
            if isinstance(child, FlashPanel):
                if child.active and child.freq == freq:
                    self.children[i] = FreqSplit(child.rect, child.depth)
                    for j, sib in enumerate(self.children):
                        if j != i:
                            sib.deactivate_tree()
                    return True
            elif isinstance(child, FreqSplit):
                if child.active and child.subdivide_matching(freq):
                    return True
        return False


def _drain_queue(q: queue.Queue[tuple[str, object]]) -> list[tuple[str, object]]:
    out: list[tuple[str, object]] = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            break
    return out


def run_pygame_loop(
    *,
    use_keyboard_injections: bool,
    outbound: queue.Queue[tuple[str, object]],
    pipeline: EegPipeline | None,
    subdivision_threshold: int,
    width_px: int,
    height_px: int,
    keyboard_only_calib_frames: int,
) -> None:
    pygame.init()
    pygame.display.set_caption("SSVEP + BCI — q quit, r reset EEG calibration + layout")
    screen = pygame.display.set_mode((width_px, height_px))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("menlo", HUD_PRIMARY_FONT_SIZE_PX)
    small = pygame.font.SysFont("menlo", HUD_SECONDARY_FONT_SIZE_PX)

    root: FreqSplit = FreqSplit((0, 0, width_px, height_px), depth=0)
    subdivision_hits = {f: 0 for f in SSVEP_FREQUENCIES_HZ}

    keyboard_calib_counter = 0
    if use_keyboard_injections:
        ui_calibrated = keyboard_only_calib_frames <= 0
    else:
        ui_calibrated = False

    running = True
    while running:
        now_ms = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r:
                    root = FreqSplit((0, 0, width_px, height_px), depth=0)
                    subdivision_hits = {f: 0 for f in SSVEP_FREQUENCIES_HZ}
                    keyboard_calib_counter = 0
                    if pipeline is not None:
                        pipeline.reset_calibration()
                    if use_keyboard_injections:
                        ui_calibrated = keyboard_only_calib_frames <= 0
                    else:
                        ui_calibrated = False

        if use_keyboard_injections and not ui_calibrated:
            keyboard_calib_counter += 1
            if keyboard_calib_counter >= keyboard_only_calib_frames:
                ui_calibrated = True

        for kind, payload in _drain_queue(outbound):
            if kind == "eeg_ready" and pipeline is not None:
                ui_calibrated = True
            elif kind == "detection":
                freq = int(payload)
                if ui_calibrated:
                    subdivision_hits[freq] = subdivision_hits.get(freq, 0) + 1

        if ui_calibrated:
            keys = pygame.key.get_pressed()
            if use_keyboard_injections:
                if keys[pygame.K_1]:
                    subdivision_hits[SSVEP_FREQUENCIES_HZ[0]] += 1
                if keys[pygame.K_2]:
                    subdivision_hits[SSVEP_FREQUENCIES_HZ[1]] += 1

            for f in SSVEP_FREQUENCIES_HZ:
                if subdivision_hits[f] >= subdivision_threshold:
                    if root.subdivide_matching(f):
                        subdivision_hits[f] = 0

        screen.fill(BACKGROUND_GRAY)
        root.update(now_ms)
        root.draw(screen, font)

        if use_keyboard_injections and not ui_calibrated:
            denom = max(1, keyboard_only_calib_frames)
            pct = int(min(100, (keyboard_calib_counter / float(denom)) * 100.0))
            hud = [
                f"Keyboard-mode calibration… {pct}% (purely visual timing; no EEG)",
                "Then 1 / 2 inject fake hits.   r = reset   q = quit",
            ]
        elif not ui_calibrated:
            if pipeline is not None:
                with pipeline._lock:
                    calib_progress = pipeline.samples_counted
            else:
                calib_progress = 0
            pct = int(min(100, (calib_progress / float(CALIBRATION_FULL_WINDOWS)) * 100.0))
            hud = [
                f"EEG calibrating… {pct}%  ({calib_progress}/{CALIBRATION_FULL_WINDOWS} windows)",
                f"Look away from flicker; after baseline, detections fill to {subdivision_threshold} to subdivide.",
            ]
        else:
            mode = "keyboard inject" if use_keyboard_injections else "OpenBCI"
            hud = [
                f"Ready ({mode}) — subdivision counters: "
                + "  ".join(f"{f} Hz={subdivision_hits[f]}" for f in SSVEP_FREQUENCIES_HZ)
                + f"  (need {subdivision_threshold} for subdivision)",
                "r = reset calibration + layout   q = quit",
            ]

        y = height_px - HUD_BOTTOM_BLOCK_HEIGHT_PX
        for line in hud:
            surf = small.render(line, True, HUD_TEXT_COLOR)
            screen.blit(surf, (HUD_MARGIN_LEFT_PX, y))
            y += HUD_LINE_STEP_PX

        pygame.display.flip()
        clock.tick(TARGET_FPS)

    pygame.quit()


def _open_board(port: str) -> OpenBCICyton:
    return OpenBCICyton(port=port)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SSVEP grid + OpenBCI CCA (guess.py-style) / keyboard test")
    p.add_argument(
        "--subdivision-hits",
        type=int,
        default=SUBDIVISION_HIT_THRESHOLD,
        help="CCA detections per frequency before subdividing that region",
    )
    p.add_argument("--width", type=int, default=WINDOW_WIDTH_PX)
    p.add_argument("--height", type=int, default=WINDOW_HEIGHT_PX)
    p.add_argument(
        "--port",
        type=str,
        default=None,
        help=(
            f"Cyton serial port (default: environment {ENV_OPENBCI_PORT_KEY!r} "
            f"or {DEFAULT_OPENBCI_PORT!r})"
        ),
    )
    p.add_argument(
        "--keyboard-only",
        action="store_true",
        help="Do not open OpenBCI; use timed fake calibration and keys 1/2 for fake hits",
    )
    p.add_argument(
        "--keyboard-calibration-frames",
        type=int,
        default=KEYBOARD_ONLY_DEFAULT_CALIBRATION_FRAMES,
        help="Frames to wait in --keyboard-only before enabling fake hits",
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    import os

    args = parse_args(argv)
    if args.subdivision_hits < 1:
        print("--subdivision-hits must be >= 1", file=sys.stderr)
        return 2
    if args.width < MIN_WINDOW_DIMENSION_PX or args.height < MIN_WINDOW_DIMENSION_PX:
        print("window too small", file=sys.stderr)
        return 2

    outbound: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=MAIN_THREAD_EVENT_QUEUE_MAXSIZE)

    if args.keyboard_only:
        run_pygame_loop(
            use_keyboard_injections=True,
            outbound=outbound,
            pipeline=None,
            subdivision_threshold=args.subdivision_hits,
            width_px=args.width,
            height_px=args.height,
            keyboard_only_calib_frames=args.keyboard_calibration_frames,
        )
        return 0

    port = args.port or os.environ.get(ENV_OPENBCI_PORT_KEY, DEFAULT_OPENBCI_PORT)
    try:
        board = _open_board(port)
    except Exception as exc:  # noqa: BLE001 — surface connection errors clearly
        print(f"Unable to connect on {port!r}: {exc}", file=sys.stderr)
        print("Set OPENBCI_PORT or pass --port, or use --keyboard-only.", file=sys.stderr)
        return 1

    print(f"Connected to OpenBCI Cyton on {port!r}.", file=sys.stderr)
    pipeline = EegPipeline(outbound)

    stream_error: list[BaseException | None] = [None]

    def stream_worker() -> None:
        try:
            board.start_stream(pipeline.process_sample)
        except BaseException as exc:  # noqa: BLE001
            stream_error[0] = exc

    worker = threading.Thread(target=stream_worker, daemon=True)
    worker.start()

    try:
        run_pygame_loop(
            use_keyboard_injections=False,
            outbound=outbound,
            pipeline=pipeline,
            subdivision_threshold=args.subdivision_hits,
            width_px=args.width,
            height_px=args.height,
            keyboard_only_calib_frames=0,
        )
    finally:
        try:
            board.stop_stream()
        except Exception:  # noqa: BLE001
            pass

    if stream_error[0] is not None:
        print(f"Stream thread ended with error: {stream_error[0]!r}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

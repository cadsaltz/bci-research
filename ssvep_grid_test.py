"""
Interactive test for a two-frequency SSVEP grid (16 Hz | 31 Hz).

Layout:
  - Root splits the window vertically: left = 16 Hz, right = 31 Hz.
  - After HIT_THRESHOLD cumulative detections for one frequency, the active
    panel for that frequency subdivides into 16 Hz and 31 Hz with the split
    rotated 90° from the parent (vertical -> horizontal -> vertical -> ...).
  - The sibling region at that level is deactivated (gray, stops flashing),
    matching the idea in demo.py.

Simulated hits (no OpenBCI required):
  - 1 / 2  : add one hit for 16 Hz / 31 Hz
  - Hold 1/2 : repeat hits each frame (fast fill toward threshold)
  - r      : reset hit counters and restore the root layout
  - q / ESC: quit

Lower the threshold for a quick UI check, e.g.:
  python ssvep_grid_test.py --threshold 5
"""

from __future__ import annotations

import argparse
import sys

import pygame


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SSVEP two-frequency grid subdivision test")
    p.add_argument(
        "--threshold",
        type=int,
        default=200,
        help="Detection hits required before subdividing the matching region (default: 200)",
    )
    p.add_argument("--width", type=int, default=1200)
    p.add_argument("--height", type=int, default=800)
    return p.parse_args(argv)


class FlashPanel:
    """Single region flashing at one frequency."""

    def __init__(self, rect: tuple[int, int, int, int], freq: int, depth: int = 0) -> None:
        self.rect = pygame.Rect(rect)
        self.freq = int(freq)
        self.depth = depth
        self.interval_ms = 1000.0 / (self.freq * 2.0)
        self.state = True
        self.last_toggle = pygame.time.get_ticks()
        self.active = True

    def deactivate_tree(self) -> None:
        self.active = False

    def update(self, now: int) -> None:
        if not self.active:
            return
        if now - self.last_toggle >= self.interval_ms:
            self.state = not self.state
            self.last_toggle = now

    def draw(self, surface: pygame.Surface, font: pygame.font.Font | None) -> None:
        if not self.active:
            color = (128, 128, 128)
        else:
            color = (0, 0, 0) if self.state else (255, 255, 255)
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, (40, 40, 40), self.rect, 1)
        if font is not None and self.active:
            label = font.render(f"{self.freq} Hz", True, (180, 180, 255))
            surface.blit(label, (self.rect.x + 8, self.rect.y + 6))


class FreqSplit:
    """Two children: first = 16 Hz side, second = 31 Hz side."""

    def __init__(self, rect: tuple[int, int, int, int], depth: int = 0) -> None:
        self.rect = pygame.Rect(rect)
        self.depth = depth
        self.active = True
        # Alternate orientation so each subdivision is rotated 90° from the parent.
        self.vertical = depth % 2 == 0
        x, y, w, h = self.rect
        d_child = depth + 1
        if self.vertical:
            w2 = w // 2
            self.children: list[FlashPanel | FreqSplit] = [
                FlashPanel((x, y, w2, h), 16, d_child),
                FlashPanel((x + w2, y, w - w2, h), 31, d_child),
            ]
        else:
            h2 = h // 2
            self.children = [
                FlashPanel((x, y, w, h2), 16, d_child),
                FlashPanel((x, y + h2, w, h - h2), 31, d_child),
            ]

    def deactivate_tree(self) -> None:
        self.active = False
        for c in self.children:
            c.deactivate_tree()

    def update(self, now: int) -> None:
        if not self.active:
            return
        for c in self.children:
            c.update(now)

    def draw(self, surface: pygame.Surface, font: pygame.font.Font | None) -> None:
        for c in self.children:
            c.draw(surface, font)

    def subdivide_matching(self, freq: int) -> bool:
        """Replace the active panel for `freq` with a nested FreqSplit; gray out sibling."""
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


def run_ui(threshold: int, width: int, height: int) -> None:
    pygame.init()
    pygame.display.set_caption("SSVEP grid test — 1/2 add hits, r reset, threshold subdivides")
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("menlo", 22)
    small = pygame.font.SysFont("menlo", 16)

    root: FreqSplit = FreqSplit((0, 0, width, height), depth=0)
    hits = {16: 0, 31: 0}
    running = True

    while running:
        now = pygame.time.get_ticks()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r:
                    hits = {16: 0, 31: 0}
                    root = FreqSplit((0, 0, width, height), depth=0)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_1]:
            hits[16] += 1
        if keys[pygame.K_2]:
            hits[31] += 1

        for f in (16, 31):
            if hits[f] >= threshold:
                if root.subdivide_matching(f):
                    hits[f] = 0

        screen.fill((90, 90, 90))
        root.update(now)
        root.draw(screen, font)

        hud_lines = [
            f"Hits: 16 Hz = {hits[16]}  |  31 Hz = {hits[31]}  (threshold = {threshold})",
            "Keys: 1 / 2 = add hit (hold for rapid)   r = reset   q = quit",
        ]
        y0 = height - 52
        for line in hud_lines:
            surf = small.render(line, True, (240, 240, 240))
            screen.blit(surf, (12, y0))
            y0 += 22

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.threshold < 1:
        print("threshold must be >= 1", file=sys.stderr)
        return 2
    run_ui(args.threshold, args.width, args.height)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

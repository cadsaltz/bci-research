"""
Run demo.py (pygame, main thread) and guess.py OpenBCI stream (background thread).

EEG detections enqueue SSVEP frequencies (Hz); the UI drains the queue and calls
subdivide(freq=...) — same as wiring described in demo.py, without merging the two modules.

  OPENBCI_PORT=/dev/cu.usbserial-XXXX python run_guess_demo.py
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading

import demo
import guess


def _stream_worker() -> None:
	assert guess.board is not None
	guess.board.start_stream(guess.process_sample)


def main(argv: list[str] | None = None) -> int:
	p = argparse.ArgumentParser(description="guess.py stream + demo.py UI (threaded queue)")
	p.add_argument(
		"--port",
		default=None,
		help="Cyton serial port (default: OPENBCI_PORT env or guess.py default)",
	)
	args = p.parse_args(argv)

	port = args.port or os.environ.get("OPENBCI_PORT", "/dev/ttyUSB0")

	outbound: queue.Queue[int] = queue.Queue(maxsize=256)
	guess.set_outbound_queue(outbound)

	try:
		guess.connect_board(port)
	except Exception as exc:  # noqa: BLE001
		print(f"Unable to connect on {port!r}: {exc}", file=sys.stderr)
		print("Set OPENBCI_PORT or pass --port.", file=sys.stderr)
		return 1

	worker = threading.Thread(target=_stream_worker, name="OpenBCI", daemon=True)
	worker.start()

	try:
		demo.main(outbound)
	finally:
		guess.stop_board()

	return 0


if __name__ == "__main__":
	raise SystemExit(main(sys.argv[1:]))

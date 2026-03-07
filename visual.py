
import sys
import numpy as np
import matplotlib.pyplot as plt
from pyOpenBCI import OpenBCICyton

fs = 250
window_size = 500
channels_to_plot = [0, 1, 2]   # O1, O2, Oz
candidate_freqs = [6, 8, 10, 12]

buffer = np.zeros((8, window_size))

# --- Setup plotting ---
plt.ion()
fig, axes = plt.subplots(7, 1, figsize=(8, 12))

# EEG line handles
eeg_lines = []
for i, ch in enumerate(channels_to_plot):
	line, = axes[i].plot(buffer[ch] - np.mean(buffer[ch], axis=0))
	axes[i].set_ylim(-200, 200)
	axes[i].set_title(f"EEG Channel {ch}")
	eeg_lines.append(line)

# Pure reference wave lines
ref_lines = []
t = np.arange(window_size) / fs

for i, f in enumerate(candidate_freqs):
	pure = np.cos(2*np.pi*f*t)
	line, = axes[i+3].plot(pure)
	axes[i+3].set_ylim(-1.5, 1.5)
	axes[i+3].set_title(f"Pure {f} Hz")
	ref_lines.append(line)

plt.tight_layout()
plt.show()

# --- Streaming function ---
def process_sample(sample):
	global buffer

	buffer[:] = np.roll(buffer, -1, axis=1)
	buffer[:, -1] = sample.channels_data

	# Update EEG plots
	for i, ch in enumerate(channels_to_plot):
 		eeg_lines[i].set_ydata(buffer[ch] - np.mean(buffer[ch], axis=0))

	plt.pause(0.001)

# --- Start board ---
try:
	board = OpenBCICyton(port='/dev/ttyUSB0')
	print("Starting stream")
except Exception as e:
	print("Unable to connect to board")
	sys.exit(1)

try:
	board.start_stream(process_sample)
except KeyboardInterrupt:
	board.stop_stream()

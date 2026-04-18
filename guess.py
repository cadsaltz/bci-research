from __future__ import annotations

import itertools
import queue
import sys
import threading
import numpy as np
from pyOpenBCI import OpenBCICyton
from sklearn.cross_decomposition import CCA
from scipy.signal import butter, filtfilt

spinner = itertools.cycle(["⠁  ", "⠃  ", "⠇  ", "⡇  ", "⣇  ", "⣧  ", "⣷  ", "⣿  ", "⣾  ", "⣼  ", "⣸  ", "⢸  ", "⠸  ", "⠘  ", "⠈  ", "   "])
spin = next(spinner)

# -------------------- PARAMETERS --------------------
fs = 250                 # Sampling rate (Hz)
window_size = 250        # Samples in one analysis window
channels_to_use = [1, 2, 3, 4, 5, 6]  # EEG channels to include in X
frequency = [16, 31]            # Frequency to test (Hz)
thresholds = {16: 0.11, 31: 0.10}
observed_freq = None

frequency_distribution = {}

for freq in frequency:
	frequency_distribution[freq] = 0

calibration_limit = 500  # Number of samples to record baseline
samples_counted = 0  # Keep track of current samples
is_calibrated = False  # Are we calibrated??
static_baseline = np.zeros(len(frequency))  # This will be our baseline once calibrated
all_calib_corrs = []  # Temporary list to store correlations during lock-in
threshold = 0.1  # Threshold for a "reading"

# Optional: set from run_guess_demo — main thread (e.g. pygame) consumes int Hz values
outbound_queue: queue.Queue[int] | None = None

_pipeline_lock = threading.Lock()

# -------------------- BUFFER --------------------
buffer = np.zeros((8, window_size))  # 8 channels × window_size

# -------------------- REFERENCE WAVE --------------------
t = np.arange(window_size) / fs
# Y matrix = sine + cosine

Y_ref = []
for freq in frequency:
	Y = []
	for k in range(1, 3):
		Y.append(np.sin(2 * np.pi * k * freq * t))
		Y.append(np.cos(2 * np.pi * k * freq * t))
	Y = np.stack(Y, axis=1)
	Y_ref.append(Y)

# -------------------- CCA SETUP --------------------
cca = CCA(n_components=1)

# -------------------- BANDPASS FUNCTION --------------------


def bandpass(data, fs, low=5, high=40, order=4):
	nyq = 0.5 * fs
	b, a = butter(order, [low / nyq, high / nyq], btype="band")  # creates the filter
	return filtfilt(b, a, data, axis=0)  # actually filters


def set_outbound_queue(q: queue.Queue[int] | None) -> None:
	"""Thread-safe way for demo/UI to receive detected SSVEP frequencies (Hz)."""
	global outbound_queue
	outbound_queue = q


# -------------------- PROCESSING FUNCTION --------------------
def process_sample(sample):
	global spin, buffer, samples_counted, is_calibrated, static_baseline, all_calib_corrs, threshold, observed_freq

	with _pipeline_lock:
		if samples_counted % 30 == 0:
			spin = next(spinner)

		# 1: Shift buffer to the left to make room for new sample
		buffer[:] = np.roll(buffer, -1, axis=1)
		# 2: Insert new sample
		buffer[:, -1] = sample.channels_data

		ch_idx = np.asarray(channels_to_use, dtype=int)
		# Only CCA channels must be non-zero (0/7 are often unused and stay 0)
		if not np.all(buffer[ch_idx, -1] != 0):
			return

		# 3: Build X matrix (time × features)
		X = buffer[channels_to_use].T  # shape: (window_size, num_channels)
		X = bandpass(X, fs)

		current_corrs = []

		# Correlation :L
		for Y in Y_ref:
			cca.fit(X, Y)
			x_c, y_c = cca.transform(X, Y)
			current_corrs.append(np.corrcoef(x_c.T, y_c.T)[0, 1])  # get correlation

		# Look at a wall - get the "average" when you're not looking at a screen
		if not is_calibrated:
			# Count samples and sample
			all_calib_corrs.append(current_corrs)
			samples_counted += 1

			if samples_counted % 25 == 0:
				print(f"Calibrating... {int((samples_counted / calibration_limit) * 100)}%")  # Keep track of calibration

			# SET UP THE BASELINE - and send a message for it!
			if samples_counted >= calibration_limit:
				static_baseline = np.mean(all_calib_corrs, axis=0)
				is_calibrated = True
				print("--- CALIBRATION COMPLETE ---")
				print(f"Baseline Noise Levels: {static_baseline}")

			return  # exit early until we're done with calbration

		# Calculate the scores based off of the static baseline
		relative_scores = np.array(current_corrs) - static_baseline  # Compare to our baseline - look at the change

		msg = f"Detected: -- Hz Observed: {observed_freq} Hz {spin}"

		for i, score in enumerate(relative_scores):
			freq = frequency[i]

			if score > thresholds[freq]:

				frequency_distribution[freq] += 1

				msg = f"Detected: {freq} Hz Observed: {observed_freq} Hz {spin}"
				if outbound_queue is not None:
					try:
						outbound_queue.put_nowait(int(freq))
					except queue.Full:
						pass
				break
		

		if samples_counted % 1000 == 0:
			
			observed_freq = max(frequency_distribution, key=frequency_distribution.get)

			for key in frequency_distribution:
				frequency_distribution[key] = 0



		sys.stdout.write("\r" + msg)
		sys.stdout.flush()

		samples_counted += 1


board: OpenBCICyton | None = None


def connect_board(port: str = "/dev/ttyUSB0") -> OpenBCICyton:
	global board
	board = OpenBCICyton(port=port)
	print("Connected to OpenBCI Cyton.")
	return board


def stop_board() -> None:
	global board
	if board is not None:
		try:
			board.stop_stream()
		except Exception:
			pass


if __name__ == "__main__":
	port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB0"
	try:
		connect_board(port)
	except Exception as e:
		print("Unable to connect to board:", e)
		sys.exit(1)

	try:
		assert board is not None
		board.start_stream(process_sample)
	except KeyboardInterrupt:
		print(f"Frequency Distribution: {frequency_distribution}")
		print("Stopping stream...")
		stop_board()

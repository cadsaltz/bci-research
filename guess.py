
import sys
import numpy as np
from pyOpenBCI import OpenBCICyton
from sklearn.cross_decomposition import CCA

# -------------------- PARAMETERS --------------------
fs = 250                 # Sampling rate (Hz)
window_size = 250        # Samples in one analysis window
channels_to_use = [6, 7, 0]  # EEG channels to include in X
frequency = [6,8,10,12]            # Frequency to test (Hz)

# -------------------- BUFFER --------------------
buffer = np.zeros((8, window_size))  # 8 channels × window_size

# -------------------- REFERENCE WAVE --------------------
t = np.arange(window_size) / fs
# Y matrix = sine + cosine

Y_ref = []
for freq in frequency:
	Y = np.stack([np.sin(2 * np.pi * freq * t), np.cos(2 * np.pi * freq * t)], axis=1)  # shape: (window_size, 2)

	Y_ref.append(Y)

# -------------------- CCA SETUP --------------------
cca = CCA(n_components=1)

# -------------------- PROCESSING FUNCTION --------------------
def process_sample(sample):
	global buffer

	# 1. Shift buffer to the left to make room for new sample
	buffer[:] = np.roll(buffer, -1, axis=1)
	# 2. Insert new sample
	buffer[:, -1] = sample.channels_data

	# Only compute CCA after buffer is full
	if np.all(buffer[:, -1] != 0):  # crude check: buffer filled
		
		# 3. Build X matrix (time × features)
		X = buffer[channels_to_use].T  # shape: (window_size, num_channels)

		# center EEG channels (mean=0)
		X = X - np.mean(X, axis=0)

		# 4. Fit CCA
		corr = [0] * 4
		for i,Y in enumerate(Y_ref):		
#			print(i)
			cca.fit(X, Y)
			x_c, y_c = cca.transform(X, Y)

			# 5. Compute canonical correlation
			corr[i] = np.corrcoef(x_c.T, y_c.T)[0, 1]
		
		#print(f"Canonical correlation: {frequency[0]}Hz: {corr[0]:.4f} | {frequency[1]}Hz: {corr[1]:.4f} | {frequency[2]}Hz: {corr[2]:.4f} | {frequency[3]}Hz: {corr[3]:.4f}")
		print(corr.index(max(corr)) * 2 + 6)

# -------------------- CONNECT TO BOARD --------------------
try:
	board = OpenBCICyton(port='/dev/ttyUSB0')
	print("Connected to OpenBCI Cyton.")
except Exception as e:
	print("Unable to connect to board:", e)
	sys.exit(1)

try:
	board.start_stream(process_sample)
except KeyboardInterrupt:
	print("Stopping stream...")
	board.stop_stream()

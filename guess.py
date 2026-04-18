import itertools
import sys
import numpy as np
from pyOpenBCI import OpenBCICyton
from sklearn.cross_decomposition import CCA
from scipy.signal import butter, filtfilt
import shared_state

spinner = itertools.cycle(["⠁  ", "⠃  ", "⠇  ", "⡇  ", "⣇  ", "⣧  ", "⣷  ", "⣿  ", "⣾  ", "⣼  ", "⣸  ", "⢸  ", "⠸  ", "⠘  ", "⠈  ", "   "])

# -------------------- PARAMETERS --------------------
fs = 250                 # Sampling rate (Hz)
window_size = 250        # Samples in one analysis window
channels_to_use = [1, 2, 3, 4, 5, 6]  # EEG channels to include in X
frequency = [16, 31]            # Frequency to test (Hz)
thresholds = {16:0.11, 31:0.10}

frequency_distribution = {}

for freq in frequency:
	frequency_distribution[freq] = 0


calibration_limit = 500  # Number of samples to record baseline
samples_counted = 0 # Keep track of current samples
is_calibrated = False # Are we calibrated??
static_baseline = np.zeros(len(frequency)) # This will be our baseline once calibrated
all_calib_corrs = [] # Temporary list to store correlations during lock-in
threshold = 0.1 #Threshold for a "reading"

# What the heck is Butterworth Bandpass Filter??? - we could look into this supposedly
# it could help

#TIME
windows_since_trigger = 0
trigger_interval = 1000

# -------------------- BUFFER --------------------
buffer = np.zeros((8, window_size))  # 8 channels × window_size

# -------------------- REFERENCE WAVE --------------------
t = np.arange(window_size) / fs
# Y matrix = sine + cosine

Y_ref = []
for freq in frequency:
	Y = []
	for k in range(1,3):
		Y.append(np.sin(2 * np.pi * k * freq * t))
		Y.append(np.cos(2 * np.pi * k * freq * t))
	Y = np.stack(Y, axis=1)
	Y_ref.append(Y)

#print("hi")
# -------------------- CCA SETUP --------------------
cca = CCA(n_components=1)


# -------------------- BANDPASS FUNCTION --------------------

def bandpass(data, fs, low=5, high=40,order=4):
	nyq = 0.5 * fs
	b, a = butter(order, [low/nyq, high/nyq], btype = 'band') #creates the filter
	return filtfilt(b, a, data, axis=0) # actually filters


# -------------------- PROCESSING FUNCTION --------------------
def process_sample(sample):
	global spin, buffer, samples_counted, is_calibrated, static_baseline, all_calib_corrs, threshold, windows_since_trigger, trigger_interval

	if samples_counted % 30 == 0:
		spin = next(spinner)

	# 1. Shift buffer to the left to make room for new sample
	buffer[:] = np.roll(buffer, -1, axis=1)
	# 2. Insert new sample
	buffer[:, -1] = sample.channels_data

	# Only compute CCA after buffer is full
	if np.all(buffer[:, -1] != 0):  
		
		# 3. Build X matrix (time × features)
		X = buffer[channels_to_use].T  # shape: (window_size, num_channels)
		# center EEG channels (mean=0)
		X = bandpass(X, fs)

		current_corrs = []

		# Correlation :L
		for Y in Y_ref:
			cca.fit(X,Y)
			x_c, y_c = cca.transform(X, Y)
			current_corrs.append(np.corrcoef(x_c.T, y_c.T)[0, 1]) #get correlation
		
		#Look at a wall - get the "average" when you're not looking at a screen
		if not is_calibrated:
			# Count samples and sample
			all_calib_corrs.append(current_corrs)
			samples_counted += 1

			if samples_counted % 25 == 0:
				print(f"Calibrating... {int((samples_counted/calibration_limit)*100)}%") # Keep track of calibration

			#SET UP THE BASELINE - and send a message for it!
			if samples_counted >= calibration_limit:
				static_baseline = np.mean(all_calib_corrs, axis=0)
				is_calibrated = True
				print("--- CALIBRATION COMPLETE ---")
				print(f"Baseline Noise Levels: {static_baseline}")
			
			return #exit early until we're done with calbration

		# Calculate the scores based off of the static baseline
		relative_scores = np.array(current_corrs) - static_baseline #Compare to our baseline - look at the change
		windows_since_trigger += 1
		print(windows_since_trigger)
		#print(relative_scores)

		msg = f"Detected: -- Hz {spin}"
		
		for i, score in enumerate(relative_scores):
			freq = frequency[i]
			if score > thresholds[freq]:
				frequency_distribution[freq] += 1
				msg = f"Detected: {freq} Hz {spin}"
				break

		if windows_since_trigger >= trigger_interval:
			windows_since_trigger = 0
			shared_state.observed_freq = frequency[np.argmax(relative_scores)]
			shared_state.new_trigger = True
			for key in frequency_distribution:
				frequency_distribution[key] = 0

			

		"""
		#Only print if the max is above some threshold - we can change this number
		if np.max(relative_scores) > threshold:
			best_freq_imp = np.argmax(relative_scores)

			frequency_distribution[frequency[best_freq_imp]] += 1

			msg = f"Detected: {frequency[best_freq_imp]} Hz {spin}"
			

		else:
			msg = f"Detected: -- Hz {spin}"
		"""

		sys.stdout.write("\r" + msg)
		sys.stdout.flush()

		samples_counted += 1
			
		'''
		#OLD CODE!!!!
		# 4. Fit CCA
		corr = [0] * 4
		for i,Y in enumerate(Y_ref):		
			#print(i)
			cca.fit(X, Y)
			x_c, y_c = cca.transform(X, Y)

			# 5. Compute canonical correlation
			corr[i] = np.corrcoef(x_c.T, y_c.T)[0, 1]
		
		#print(f"Canonical correlation: {frequency[0]}Hz: {corr[0]:.4f} | {frequency[1]}Hz: {corr[1]:.4f} | {frequency[2]}Hz: {corr[2]:.4f} | {frequency[3]}Hz: {corr[3]:.4f}")
		print(corr.index(max(corr)) * 2 + 6)
		'''

# -------------------- CONNECT TO BOARD --------------------
def start_eeg():
	try:
		board = OpenBCICyton(port='/dev/cu.usbserial-D200QSOE')
		#board = OpenBCICyton(port='/dev/ttyUSB0')
		print("Connected to OpenBCI Cyton.")
	except Exception as e:
		print("Unable to connect to board:", e)
		

	try:
		board.start_stream(process_sample)
	except KeyboardInterrupt:
		print(f"Frequency Distribution: {frequency_distribution}")
		print("Stopping stream...")
		board.stop_stream()

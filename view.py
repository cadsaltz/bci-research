
#!/usr/bin/env python3

"""

"""

import argparse
import sys
from pathlib import Path		#arg parsing 
import mne				#edf file reading
import matplotlib.pyplot as plt		#plotting signal
import numpy as np			#fast fourier transformation

def parse_args():

	parser = argparse.ArgumentParser(
		description="Processing FFT of EEG data over a fixed duration."
	)

	parser.add_argument(
		"input_file",
		type=Path,
		help="Path to EDF file containing EEG data"
	)

	parser.add_argument(
		"--duration",
		type=float,
		default=None,
		help="Duration (in seconds) of data to use for FFT"
	)

	return parser.parse_args()


class WaveForm:

	"""
	Encapsulates a single EEG channel waveform from an MNE raw object
	automatically selects the Oz.. EEG channel

	Attributes:
		raw: the full mne.io.Raw object
		signal: 1D numpy array of EEG amplitudes
		times: 1D numpy array of times in seconds
		fs: sampling rate in Hz
		fft_vals: FFT amplitudes
		freqs: corresponding frequency axis (Hz)

	"""

	def __init__(self, raw: mne.io.BaseRaw):
		self.raw = raw

		# extract signal and time axis
		data, times = raw.get_data(
			picks=["Oz.."],
			return_times=True
		)

		# save fields
		self.signal = data[0]
		self.times = times
		self.fs = raw.info["sfreq"]

		# placeholders for "expensive" computation
		self.fft_vals = None
		self.freqs = None


	# plot the waveform with optional fft window highlighted
	def plot_eeg(self, duration: float = None):
		
		# appearance
		plt.figure(figsize=(12,4))
		plt.plot(self.times, self.signal, color="skyblue", label="EEG Signal")

		# highlighting window
		if duration is not None:
			n_samples = int(duration * self.fs)

			plt.axvspan(self.times[0], self.times[n_samples - 1],
				color="orange", alpha=0.3, label=f"{duration}s window")

		plt.xlabel("Time (s)")
		plt.ylabel("Aplitude (micro V)")
		plt.title("EEG Signal")
		plt.legend()
		plt.tight_layout()
		plt.show(block=False)

	# compute the fft values of the eeg signal
	def compute_fft(self, duration: float = None):

		# only comput the fft of the give window
		if duration is not None:
			n_samples = int(duration * self.fs)
			signal = self.signal[:n_samples]

		# default to the entire signal
		else:
			signal = self.signal

		# compute fft values
		number_of_samples = len(signal)
		fft_vals = np.fft.rfft(signal)
		freqs = np.fft.rfftfreq(number_of_samples, d= 1/self.fs)

		# convert complex fft values to magnitudes
		self.fft_vals = np.abs(fft_vals) / number_of_samples
		self.freqs = freqs

	# plot the fft of the waveform
	def plot_fft(self, max_freq: float = 50):
		
		# must compute the fft values to plot them
		if self.fft_vals is None or self.freqs is None:
			raise RuntimeError("FFT not computed yet. Call compute_fft()")

		plt.figure(figsize=(10,4))
		plt.plot(self.freqs, self.fft_vals, color="purple")
		plt.xlabel("frequency (Hz)")
		plt.ylabel("Amplitude (micro V)")
		plt.title("FFT")
		plt.xlim(0, max_freq)
		plt.tight_layout()
		plt.show(block=False)

# ingest raw eeg data from a file
def ingest_data(input_file):

	if not input_file.exists():
		print(f"File does not exist: {input_file}")
		sys.exit(1)
	
	return mne.io.read_raw_edf(input_file, preload=True)



def main():

	args = parse_args()

	input_file = args.input_file
	duration = args.duration

	raw = ingest_data(input_file)

	print(raw)
	#print(raw.ch_names)

	wave = WaveForm(raw)

	wave.plot_eeg(duration)

	wave.compute_fft(duration)

	wave.plot_fft()


	input("Press enter to exit")
	plt.close("all")



if __name__ == "__main__":
	main()

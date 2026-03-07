
import math
import numpy as np
import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')
from pyOpenBCI import OpenBCICyton

fs = 250
window_size = 500
num_channels = 8
buffer = np.zeros((num_channels, window_size))

sample_count = 0
update_every = window_size

def process_sample(sample):
	global buffer, sample_count
	buffer = np.roll(buffer, -1, axis=1)
	buffer[:, -1] = sample.channels_data

	#print(buffer)
	#x = buffer[0]
	#y = range(window_size)
	#fig, ax = plt.subplots()

	if sample_count == window_size:
		sample_count = 0
		#ax.plot(buffer[0])
	else:
		sample_count += 1



# Connect to your Cyton
board = OpenBCICyton(port='/dev/ttyUSB0')

try:
	board.start_stream(process_sample)
except KeyboardInterrupt:
	print("Stopping stream...")
finally:
	board.stop_stream()
	print(buffer)
	x1 = np.array(buffer[0])

#	fig, ax = plt.subplots()


	plt.figure()
	plt.plot(np.arange(len(x1)),x1)
	plt.show()


	i = np.arange(window_size)
	x2 = np.cos(2*np.pi*6*i/250)

	plt.figure()
	plt.plot(np.arange(len(x2)), x2)
	plt.show()
	print(math.pi)



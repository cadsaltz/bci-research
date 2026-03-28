import time
import numpy as np
import matplotlib.pyplot as plt
from pyOpenBCI import OpenBCICyton

# ---------------- PARAMETERS ----------------
fs = 250                 # Cyton sampling rate (Hz)
window_size = 250        # samples in the buffer (~1 second)
VAR_THRESHOLD = 5e-6     # ignore tiny noise

# ---------------- BUFFER -------------------
buffer = np.zeros((8, window_size))  # 8 channels x window_size

# ---------------- CONNECT TO BOARD ----------------
PORT = '/dev/cu.usbserial-D200QSOE'  # <-- replace with your correct port
try:
    board = OpenBCICyton(port=PORT)
    print("Connected to OpenBCI Cyton.")
except Exception as e:
    print("Unable to connect to board:", e)
    exit(1)

# ---------------- MATPLOTLIB SETUP ----------------
plt.ion()  # interactive mode
fig, ax = plt.subplots(figsize=(10,6))
lines = []
for ch in range(8):
    line, = ax.plot(np.zeros(window_size), label=f'Ch {ch}')
    lines.append(line)
ax.set_ylim(-100, 100)  # initial scale, adjust if needed
ax.set_title("EEG Channel Std Visualizer")
ax.set_xlabel("Samples")
ax.set_ylabel("Voltage")
ax.legend()
plt.show()

# ---------------- PROCESS SAMPLE ----------------
def process_sample(sample):
    global buffer, lines

    # Shift buffer left and append new sample
    buffer[:] = np.roll(buffer, -1, axis=1)
    buffer[:, -1] = sample.channels_data

    # compute std per channel
    stds = np.std(buffer, axis=1)
    active_channels = [i for i, s in enumerate(stds) if s > VAR_THRESHOLD]

    # update matplotlib lines
    for ch, line in enumerate(lines):
        line.set_ydata(buffer[ch])
        line.set_alpha(1.0 if ch in active_channels else 0.3)

    ax.set_ylim(-max(1e-5, np.max(buffer))*1.2, max(1e-5, np.max(buffer))*1.2)
    fig.canvas.draw()
    fig.canvas.flush_events()

# ---------------- START STREAM ----------------
try:
    print("Streaming... Touch electrodes to identify channels.")
    board.start_stream(process_sample)
except KeyboardInterrupt:
    print("Stopping stream...")
    board.stop_stream()
    plt.ioff()
    plt.show()
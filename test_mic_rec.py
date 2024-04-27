import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser = argparse.ArgumentParser()
parser.add_argument(
    '-o',
    '--output',
    default='test.wav',
    help='Recorded sound file')
parser.add_argument(
    '-d',
    '--device',
    type=int,
    help='Choised record device ID')
args = parser.parse_args()

import sounddevice as sd
from scipy.io.wavfile import write

import contextlib
import os
import sys

@contextlib.contextmanager
def ignore_stderr():
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    sys.stderr.flush()
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)

if args.device:
  sd.default.device = args.device

devices = sd.query_devices()
devices1 = sd.query_devices(device=None,kind='input')

print("Audiodevices:")
print(devices)
print()
print("Selected microphone:")
print(" ", devices1["index"], devices1["name"])
print()
print("File name:")
print(" ", args.output)
print()

fs = 44100  # Sample rate
seconds = 10  # Duration of recording

sd.default.samplerate = fs
sd.default.channels = 1

#if args.device:
#  sd.default.device = args.device

with ignore_stderr():
    myrecording = sd.rec(int(seconds * fs))
    sd.wait()  # Wait until recording is finished
write(args.output, fs, myrecording)  # Save as WAV file

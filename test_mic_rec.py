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

myrecording = sd.rec(int(seconds * fs))
sd.wait()  # Wait until recording is finished
write(args.output, fs, myrecording)  # Save as WAV file

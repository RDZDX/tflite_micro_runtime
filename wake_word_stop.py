"""
Whenever you say "stop", get text "stop" on console output
"""

import sounddevice as sd
import numpy as np
import scipy.signal
import timeit
import python_speech_features
import sys
import contextlib
import os

sys.tracebacklimit=0

devices = sd.query_devices()
devices1 = sd.query_devices(device=None,kind='input')
print("Audiodevices:")
print(devices)
print()
print("Microphone:")
print(" ", devices1["index"], devices1["name"])
print()

@contextlib.contextmanager # for hidding bluealsa log output
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

#from tflite_runtime.interpreter import Interpreter
from tflite_micro_runtime.interpreter import Interpreter

# Parameters
debug_time = 1
debug_acc = 0
#word_threshold = 0.5
word_threshold = 0.74
#rec_duration = 0.5
rec_duration = 1 
#window_stride = 0.5
window_stride = 1
sample_rate = 48000
resample_rate = 8000
num_channels = 1
num_mfcc = 16

model_path = 'wake_word_stop_lite.tflite'

# Sliding window
window = np.zeros(int(rec_duration * resample_rate) * 2)

#print(window)
#print(int(rec_duration * resample_rate) * 2)

# Load model (interpreter)
interpreter = Interpreter(model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
#print(input_details)
#print()
#print(output_details)

# Decimate (filter and downsample)
def decimate(signal, old_fs, new_fs):
    
    # Check to make sure we're downsampling
    if new_fs > old_fs:
        print("Error: target sample rate higher than original")
        return signal, old_fs
    
    # We can only downsample by an integer factor
    dec_factor = old_fs / new_fs
    if not dec_factor.is_integer():
        print("Error: can only decimate by integer factor")
        return signal, old_fs

    # Do decimation
    resampled_signal = scipy.signal.decimate(signal, int(dec_factor))

    return resampled_signal, new_fs

# This gets called every 0.5 seconds
def sd_callback(rec, frames, time, status):

    # Start timing for testing
    start = timeit.default_timer()
    
    # Notify if errors
    if status:
       print('Error:', status)
    
#    print(rec.shape)

    # Remove 2nd dimension from recording sample
    rec = np.squeeze(rec)

    # Resample
    rec, new_fs = decimate(rec, sample_rate, resample_rate)

#    print(rec.shape)

    # Save recording onto sliding window

    window[:len(window)//2] = window[len(window)//2:]
    window[len(window)//2:] = rec

#    print(rec.shape)

    # Compute features
    mfccs = python_speech_features.base.mfcc(window, 
                                        samplerate=new_fs,
                                        winlen=0.512,
                                        winstep=0.100,
                                        numcep=num_mfcc,
                                        nfilt=26,
                                        nfft=2048,
                                        preemph=0.0,
                                        ceplifter=0,
                                        appendEnergy=False,
                                        winfunc=np.hanning)

#                                       winlen=0.256,
#                                       winstep=0.050,

    mfccs = mfccs.transpose()
#    print(mfccs.shape)

    # Make prediction from model
    in_tensor = np.float32(mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1))
#    print(in_tensor.shape)
    interpreter.set_tensor(input_details[0]['index'], in_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    val = output_data[0][0]
    if val > word_threshold:
        print('stop')

    if debug_acc:
        print(val)
    
    if debug_time:
        print(timeit.default_timer() - start)

# Start streaming from microphone

with ignore_stderr():
    with sd.InputStream(channels=num_channels,
                        samplerate=sample_rate,
                        blocksize=int(sample_rate * rec_duration),
                        callback=sd_callback):
        while True:
            pass

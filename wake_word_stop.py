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
#debug_time = 0
debug_acc = 0
#debug_acc = 1
#word_threshold = 0.5
word_threshold = 0.74

#rec_duration = 0.25
#rec_duration = 0.50
rec_duration = 0.75
#rec_duration = 1.00
#rec_duration = 1.25
#rec_duration = 1.50
#rec_duration = 1.75
#rec_duration = 2.00

#window_stride = 0.25
window_stride = 0.50
#window_stride = 0.75
#window_stride = 1.00
#window_stride = 1.25
#window_stride = 1.50
#window_stride = 1.75
#window_stride = 2.00

sample_rate = 48000
resample_rate = 8000
num_channels = 1
num_mfcc = 16

model_path = 'wake_word_stop_lite.tflite'
#model_path = 'audio_preprocessor_float.tflite' # - bad
#model_path = 'non_stream.tflite'
#model_path = 'stream_state_external.tflite'
#model_path = 'hey_mycroft_v0.1.tflite' # - uzsiveda !
#model_path = 'micro_speech_quantized.tflite' #index i3
#model_path = 'keyword_spotting_on_off.tflite'

#model_path = 'audio_preprocessor_int8.tflite'
#model_path = 'efficientnet_lite0.tflite'
#model_path = 'tflite_micro_speech.tflite'
#model_path = 'trained_model.tflite'

wake_word = ['stop'] # ['start', 'stop', 'left', 'right']

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
#print()
print("Start listening:")
print()

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

#    print(int(dec_factor))

    return resampled_signal, new_fs

# This gets called every 0.5 seconds
def sd_callback(rec, frames, time, status):

    # Start timing for testing
    start = timeit.default_timer()
    
    # Notify if errors
#    if status:
#       print('Error:', status)
    
#    print(rec.shape)

    # Remove 2nd dimension from recording sample
    rec = np.squeeze(rec)

#    print(rec.shape)

    # Resample
    rec, new_fs = decimate(rec, sample_rate, resample_rate)

#    print(rec.shape)

    # Save recording onto sliding window

    window[:len(window)//2] = window[len(window)//2:]
    window[len(window)//2:] = rec

    # Compute features
    mfccs = python_speech_features.base.mfcc(window, 
                                        samplerate=new_fs,
                                        winlen=0.384,
                                        winstep=0.075,
                                        numcep=num_mfcc,
                                        nfilt=26,
                                        nfft=2048,
                                        preemph=0.0,
                                        ceplifter=0,
                                        appendEnergy=False,
                                        winfunc=np.hanning)

#                                       winlen=0.128,
#                                       winstep=0.025
#                                       nfilt=26,

#                                       winlen=0.256,
#                                       winstep=0.050,
#                                       nfilt=26,

#                                       winlen=0.384,
#                                       winstep=0.075,
#                                       nfilt=26,

#                                       winlen=0.512,
#                                       winstep=0.100,
#                                       nfilt=26,

#                                       winlen=0.640,
#                                       winstep=0.125,
#                                       nfilt=26,

#                                       winlen=0.768,
#                                       winstep=0.150,
#                                       nfilt=26,

#                                       winlen=0.896,
#                                       winstep=0.175,
#                                       nfilt=26,

#                                       winlen=1.024,
#                                       winstep=0.200,
#                                       nfilt=26,

    mfccs = mfccs.transpose()
#    print(mfccs.shape)

    # Make prediction from model
    in_tensor = np.float32(mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1))
#    print(in_tensor.shape)
    interpreter.set_tensor(input_details[0]['index'], in_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

#    val = output_data[0][0]
#    if val > word_threshold:
#        print('stop')

    prediction = output_data.argmax(axis=0)
    val = np.max(output_data)
    tes = wake_word[prediction[0]]

    if val > word_threshold:
        print(tes)

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

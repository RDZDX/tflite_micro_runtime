#!/usr/bin/python3

import argparse
import time
import numpy as np
import sys
import os
import contextlib

import tflite_micro_runtime.interpreter as tflite
from tflite_micro_runtime.image_transform import ImageTransformer

from picamera2 import Picamera2, Preview

sys.tracebacklimit = 0

# -----------------------------
# Camera sizes
# -----------------------------

normalSize = (640, 480)

# Lower resolution = faster inference
lowresSize = (320, 240)

# -----------------------------
# Arguments
# -----------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model',
    help='Path of the detection model.',
    required=True
)

parser.add_argument(
    '--label',
    help='Path of the labels file.'
)

args = parser.parse_args()

# -----------------------------
# Silence picamera2 logs
# -----------------------------

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

# -----------------------------
# Labels
# -----------------------------

def ReadLabelFile(file_path):

    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

labels = ReadLabelFile(args.label) if args.label else None

# -----------------------------
# TensorFlow Lite
# -----------------------------

interpreter = tflite.Interpreter(model_path=args.model)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

height = input_details['shape'][1]
width = input_details['shape'][2]

input_index = input_details['index']
output_index = output_details['index']

floating_model = input_details['dtype'] == np.float32

# Direct access to input tensor
input_tensor = interpreter.tensor(input_index)()[0]

print()
print("Model:", args.model)
print("Labels:", args.label)
print("Input shape:", width, "x", height)
print("Float model:", floating_model)
print()

# -----------------------------
# ImageTransformer
# -----------------------------

roi_w, roi_h = lowresSize

img_xfrm = ImageTransformer(
    src_points=[
        [0, 0],
        [roi_w, 0],
        [roi_w - 1, roi_h - 1],
        [0, roi_h - 1]
    ],
    dst_size=(width, height),

    # Standardization only for uint8 models
    standardize=not floating_model
)

# -----------------------------
# Preallocated RGB buffer
# -----------------------------

rgb_buf = np.empty(
    (roi_h, roi_w, 3),
    dtype=np.uint8
)

# -----------------------------
# Print control
# -----------------------------

frame_counter = 0
PRINT_EVERY_N_FRAMES = 10

# -----------------------------
# Inference
# -----------------------------

def InferenceTensorFlow(grey):

    global frame_counter

    # Remove stride padding
    grey = grey[:, :roi_w]

    # Fast grayscale -> RGB conversion
    rgb_buf[:, :, 0] = grey
    rgb_buf[:, :, 1] = grey
    rgb_buf[:, :, 2] = grey

    # Perspective transform + resize
    x = img_xfrm.invoke(rgb_buf)

    # Float model normalization
    if floating_model:
        x = (x.astype(np.float32) - 127.5) / 127.5

    # Copy directly into input tensor
    np.copyto(input_tensor, x)

    start = time.time()

    interpreter.invoke()

    elapsed_ms = (time.time() - start) * 1000.0

    results = np.squeeze(
        interpreter.get_tensor(output_index)
    )

    top = np.argmax(results)

    # Avoid printing every frame
    frame_counter += 1

    if frame_counter >= PRINT_EVERY_N_FRAMES:

        if labels:
            print(f'{labels[top]} {elapsed_ms:.2f}ms')
        else:
            print(f'{top} {elapsed_ms:.2f}ms')

        frame_counter = 0

# -----------------------------
# Camera
# -----------------------------

with ignore_stderr():

    picam2 = Picamera2()

    config = picam2.create_preview_configuration(

        main={
            "size": normalSize
        },

        lores={
            "size": lowresSize,
            "format": "YUV420"
        },

        controls={

            # Lower FPS reduces CPU usage

            # 10 FPS
            # "FrameDurationLimits": (100000, 100000)

            # 5 FPS
            "FrameDurationLimits": (200000, 200000)

            # 2 FPS
            # "FrameDurationLimits": (500000, 500000)
        },

        buffer_count=4
    )

    picam2.configure(config)

    stride = picam2.stream_configuration(
        "lores"
    )["stride"]

    # Disable preview for max performance !!!!!!!!!!!!!!
    picam2.start_preview(Preview.NULL)

    picam2.start()

    while True:

        buffer = picam2.capture_buffer("lores")

        grey = buffer[
            :stride * roi_h
        ].reshape((roi_h, stride))

        InferenceTensorFlow(grey)
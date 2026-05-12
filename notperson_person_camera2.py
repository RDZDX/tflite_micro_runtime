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

normalSize = (640, 480)
lowresSize = (640, 480)

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path of the detection model.', required=True)
parser.add_argument('--label', help='Path of the labels file.')
args = parser.parse_args()


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


def ReadLabelFile(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]


if args.label:
    labels = ReadLabelFile(args.label)
else:
    labels = None


interpreter = tflite.Interpreter(model_path=args.model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

input_index = input_details[0]['index']
output_index = output_details[0]['index']

floating_model = input_details[0]['dtype'] == np.float32

print()
print("Model (", args.model, ")")
print("Labels (", args.label, ")")
print("Image shape (", width, ",", height, ")")
print()

roi_w, roi_h = lowresSize

img_xfrm = ImageTransformer(
    src_points=[
        [0, 0],
        [roi_w, 0],
        [roi_w - 1, roi_h - 1],
        [0, roi_h - 1]
    ],
    dst_size=(width, height),
    standardize=True
)


def InferenceTensorFlow(image):

    # Convert grayscale Y plane to RGB
    rgb = np.stack((image,) * 3, axis=-1)

    # Perspective transform + resize + standardize
    x = img_xfrm.invoke(rgb)

    x = np.expand_dims(x, axis=0)

    if floating_model:
        x = (np.float32(x) - 127.5) / 127.5

    input_tensor = interpreter.tensor(input_index)()[0]
    np.copyto(input_tensor, x[0])

    start_time = time.time()

    interpreter.invoke()

    stop_time = time.time()

    output_data = interpreter.get_tensor(output_index)
    results = np.squeeze(output_data)

    top_k = results.argsort()[-1:][::-1]

    for i in top_k:
        print(labels[i], ' {:.3f}ms'.format((stop_time - start_time) * 1000))


with ignore_stderr():

    picam2 = Picamera2()

    picam2.start_preview(Preview.DRM)

    config = picam2.create_preview_configuration(
        main={"size": normalSize},
        lores={"size": lowresSize, "format": "YUV420"}
    )

    picam2.configure(config)

    stride = picam2.stream_configuration("lores")["stride"]

    picam2.start()

    while True:
        buffer = picam2.capture_buffer("lores")

        grey = buffer[:stride * lowresSize[1]].reshape(
            (lowresSize[1], stride)
        )

        # remove stride padding if present
        grey = grey[:, :lowresSize[0]]

        _ = InferenceTensorFlow(grey)
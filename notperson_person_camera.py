#!/usr/bin/python3


# and run from the command line,
#
# $ python3 notperson_person_camera.py --model notperson_person.tflite --label labels_notperson_person.txt

import argparse
import time
import numpy as np
import sys
import tflite_micro_runtime.interpreter as tflite
#import tflite_micro_runtime.image_transform as tfl_transf

from picamera2 import Picamera2, Preview
from PIL import Image

sys.tracebacklimit=0

normalSize = (640, 480)
lowresSize = (320, 240)

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path of the detection model.', required=True)
parser.add_argument('--label', help='Path of the labels file.')
args = parser.parse_args()

def ReadLabelFile(file_path):
    with open(file_path, 'r') as f:
      return [line.strip() for line in f.readlines()]

if (args.label):
    labels = ReadLabelFile(args.label)
else:
    labels = None

interpreter = tflite.Interpreter(model_path=args.model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
floating_model = False
if input_details[0]['dtype'] == np.float32:
    floating_model = True

print()
print("Model (",args.model, ")")
print("Labels (",args.label, ")")
print("Image shape (", width, ",", height, ")")
print()

#roi_w, roi_h = 128,128
#roi_w, roi_h = width,height

#img_xfrm = tfl_transf.ImageTransformer(
#  src_points=[[0, 0], [roi_w, 0], [roi_w-1, roi_h-1], [0, roi_h-1]],
#  dst_size=(width,height),
#  standardize=True
#)

def InferenceTensorFlow(image):

#    rgb = Image.frombuffer("L", (320, 240), image, 'raw', "L", 0, 1).convert('RGB')
    rgb = Image.frombuffer("L", (lowresSize), image, 'raw', "L", 0, 1).convert('RGB')

    picture = rgb.resize((width, height), Image.Resampling.LANCZOS)

#    x = img_xfrm.invoke(picture)
#    input_data = np.expand_dims(x, axis=0)

    input_data = np.expand_dims(picture, axis=0)
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    interpreter.set_tensor(input_details[0]['index'], input_data)

    start_time = time.time()

    interpreter.invoke()

    stop_time = time.time()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)

    top_k = results.argsort()[-1:][::-1]

    for i in top_k:
      if floating_model:
        print(labels[i], ' {:.3f}ms'.format((stop_time - start_time) * 1000))
      else:
        print(labels[i], ' {:.3f}ms'.format((stop_time - start_time) * 1000))

picam2 = Picamera2()

picam2.start_preview(Preview.DRM)
#picam2.start_preview(Preview.QTGL)
#picam2.start_preview(Preview.NULL)

config = picam2.create_preview_configuration(main={"size": normalSize},
                                                 lores={"size": lowresSize, "format": "YUV420"})
picam2.configure(config)

#picam2.set_controls({"FrameRate": 1})

stride = picam2.stream_configuration("lores")["stride"]

picam2.start()

while True:
    buffer = picam2.capture_buffer("lores")
    grey = buffer[:stride * lowresSize[1]].reshape((lowresSize[1], stride))
    _ = InferenceTensorFlow(grey)

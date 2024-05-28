#!/usr/bin/python3


# and run from the command line,
#
# $ python3 notperson_person_camera.py --model mobilenet_v2.tflite --label coco_labels.txt

import argparse
import time
import numpy as np
import sys
import tflite_micro_runtime.interpreter as tflite

from picamera2 import MappedArray, Picamera2, Preview
from PIL import Image, ImageDraw, ImageOps

sys.tracebacklimit=0

normalSize = (640, 480)
lowresSize = (320, 240)

def ReadLabelFile(file_path):
    with open(file_path, 'r') as f:
      return [line.strip() for line in f.readlines()]

def InferenceTensorFlow(image, model, output, label=None):
    global rectangles

    if label:
        labels = ReadLabelFile(label)
    else:
        labels = None

    interpreter = tflite.Interpreter(model_path=model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    floating_model = False
    if input_details[0]['dtype'] == np.float32:
        floating_model = True
   
    rgb = Image.frombuffer("L", (4, 4), image, 'raw', "L", 0, 1).convert('RGB')

    picture = rgb.resize((width, height), Image.Resampling.LANCZOS)

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
    labels = ReadLabelFile(label)

    for i in top_k:
      if floating_model:
        print(labels[i], ' {:.3f}ms'.format((stop_time - start_time) * 1000))
      else:
        print(labels[i], ' {:.3f}ms'.format((stop_time - start_time) * 1000))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path of the detection model.', required=True)
    parser.add_argument('--label', help='Path of the labels file.')
    parser.add_argument('--output', help='File path of the output image.')
    args = parser.parse_args()

    if (args.output):
        output_file = args.output
    else:
        output_file = 'out.jpg'

    if (args.label):
        label_file = args.label
    else:
        label_file = None

    picam2 = Picamera2()

    picam2.start_preview(Preview.DRM)
#   picam2.start_preview(Preview.QTGL)
#   picam2.start_preview(Preview.NULL)

    config = picam2.create_preview_configuration(main={"size": normalSize},
                                                 lores={"size": lowresSize, "format": "YUV420"})
    picam2.configure(config)

#    picam2.set_controls({"FrameRate": 1})

    stride = picam2.stream_configuration("lores")["stride"]

# for testing information display only -  this is not  necessary block !

# begin:
    interpreter = tflite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    print()
    print("Model (",args.model, ")")
    print("Labels (",args.label, ")")
#    print("Image (",args.output, ")")
    print("Image shape (", width, ",", height, ")")
# end

    print()

    picam2.start()

    while True:
        buffer = picam2.capture_buffer("lores")
        grey = buffer[:stride * lowresSize[1]].reshape((lowresSize[1], stride))
        _ = InferenceTensorFlow(grey, args.model, output_file, label_file)


if __name__ == '__main__':
    main()


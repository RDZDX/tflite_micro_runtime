#!/usr/bin/python3


# and run from the command line,
#
# $ python3 real_time.py --model notperson_person.tflite --label labels_notperson_person.txt


import argparse
import time
import numpy as np
#import tflite_runtime.interpreter as tflite
import tflite_micro_runtime.interpreter as tflite

from picamera2 import MappedArray, Picamera2, Preview

from PIL import Image, ImageDraw, ImageOps

normalSize = (640, 480)
lowresSize = (320, 240)

rectangles = []

def ReadLabelFile(file_path):
    with open(file_path, 'r') as f:
      return [line.strip() for line in f.readlines()]

#def ReadLabelFile(file_path):
#    with open(file_path, 'r') as f:
#        lines = f.readlines()
#    ret = {}
#    for line in lines:
#        pair = line.strip().split(maxsplit=1)
#        ret[int(pair[0])] = pair[1].strip()
#    return ret

def DrawRectangles(request):
    with MappedArray(request, "main") as m:
        for rect in rectangles:
            rect_start = (int(rect[0] * 2) - 5, int(rect[1] * 2) - 5)
            rect_end = (int(rect[2] * 2) + 5, int(rect[3] * 2) + 5)
#           cv2.rectangle(m.array, rect_start, rect_end, (0, 255, 0, 0))


def InferenceTensorFlow(image, model, output, label=None):
    global rectangles

    if label:
        labels = ReadLabelFile(label)
    else:
        labels = None

#    interpreter = tflite.Interpreter(model_path=model, num_threads=4)
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

#    print("Model (",args.model_file, ")")
#    print("Labels (",args.label, ")")
#    print("Image (",args.image, ")")
#    print("Image shape (", width, ",", height, ")")

    for i in top_k:
      if floating_model:
#        print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
        print('{:08.6f}: {}'.format(float(results[i]), labels[i]), ' time: {:.3f}ms'.format((stop_time - start_time) * 1000))
      else:
#        print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))
        print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]), ' time: {:.3f}ms'.format((stop_time - start_time) * 1000))

#    print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))

    detected_boxes = interpreter.get_tensor(output_details[0]['index'])
    detected_classes = interpreter.get_tensor(output_details[0]['index'])
    detected_scores = interpreter.get_tensor(output_details[0]['index'])
    num_boxes = interpreter.get_tensor(output_details[0]['index'])

    rectangles = []
#    for i in range(int(num_boxes)):
#        top, left, bottom, right = detected_boxes[0][i]
#        classId = int(detected_classes[0][i])
#        score = detected_scores[0][i]
#        if score > 0.5:
#            xmin = left * initial_w
#            ymin = bottom * initial_h
#            xmax = right * initial_w
#            ymax = top * initial_h
#            if labels:
#                print(labels[classId], 'score = ', score)
#            else:
#                print('score = ', score)
#            box = [xmin, ymin, xmax, ymax]
#           rectangles.append(box)


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
    config = picam2.create_preview_configuration(main={"size": normalSize},
                                                 lores={"size": lowresSize, "format": "YUV420"})
    picam2.configure(config)

#    picam2.set_controls({"FrameRate": 1})

    stride = picam2.stream_configuration("lores")["stride"]
    picam2.post_callback = DrawRectangles # +++++++++++++++++++++++++++++++++++++++++++++++++++

    picam2.start()

    while True:
        buffer = picam2.capture_buffer("lores")
        grey = buffer[:stride * lowresSize[1]].reshape((lowresSize[1], stride))
        _ = InferenceTensorFlow(grey, args.model, output_file, label_file)


if __name__ == '__main__':
    main()


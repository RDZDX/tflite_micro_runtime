import numpy as np
import tflite_micro_runtime.interpreter as tf
from PIL import Image
import time

def load_labels(path): # Read the labels from the text file as a Python list.
  with open(path, 'r') as f:
    return [line.strip() for i, line in enumerate(f.readlines())]

def set_input_tensor(interpreter, new_image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = new_image

def classify_image(interpreter, new_image, top_k=1):
  set_input_tensor(interpreter, new_image)

#labels = '''airplane automobile bird cat deerdog frog horseship truck'''.split()
labels = load_labels('labels_cifar_10.txt')
#labels = load_labels('labels.txt')

# Initialize the TFLite interpreter
interpreter = tf.Interpreter(model_path="cifar_10.tflite")
#interpreter = tf.Interpreter(model_path="mobilenet_v1_25.tflite")

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
#Read image
# Imports PIL module
#from PIL import Image
# open method used to open different extension image file
#im = Image.open("car.jpg")
#im = Image.open("bird1.jpg")
#im = Image.open("person.jpg")
#im = Image.open("grace_hopper.bmp")
#im = Image.open("person.png")
#im = Image.open("test_cat.jpg")
#im = Image.open("test_cat1.jpg")
#im = Image.open("test_cat2.jpg")
#im = Image.open("test_frog.jpg")
#im = Image.open("test_frog1.jpg")
#im = Image.open("test_frog2.jpg")
#im = Image.open("test_bird.jpg")
#im = Image.open("test_bird1.jpg")
#im = Image.open("test_bird2.jpg")
#im = Image.open("test_airplane.jpg")
im = Image.open("test_airplane2.bmp")

new_image = im.resize((32, 32))
x_test=np.array(new_image)
x_test = x_test/255

# Invoke the interpreter
time1 = time.time()
interpreter.set_tensor(input_details["index"], [x_test.astype(np.float32)])
interpreter.invoke()
y_pred1 = interpreter.get_tensor(output_details["index"])[0]
# save the predicted label
predicted_label1 = labels[y_pred1.argmax()]

prob = classify_image(interpreter, new_image)
time2 = time.time()
classification_time = np.round(time2-time1, 3)
print("Classification Time =", classification_time, "seconds.")
print("Predicted label is:", predicted_label1, ", with accuracy:", prob, "%.") 

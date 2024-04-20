import numpy as np
import tflite_micro_runtime.interpreter as tf
from PIL import Image
import time

def load_labels(path): # Read the labels from the text file as a Python list.
  with open(path, 'r') as f:
    return [line.strip() for i, line in enumerate(f.readlines())]

my_model_path = "cifar_10.tflite"
my_labels_path = "labels_cifar_10.txt"

#labels = '''airplane automobile bird cat deer dog frog horse ship truck'''.split()

time1 = time.time()

labels = load_labels(my_labels_path)

# Initialize the TFLite interpreter
interpreter = tf.Interpreter(model_path=my_model_path)

print("Model:", my_model_path, "loaded successfully.")
print("Labels:", my_labels_path, "loaded successfully.")

interpreter.allocate_tensors()
_, height, width, _ = interpreter.get_input_details()[0]['shape']
print("Image Shape (",width,",",height,")")
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

#Read image
# open method used to open different extension image file
new_image = Image.open("test_cat.jpg").convert('RGB').resize((width, height), Image.Resampling.LANCZOS)
x_test=np.array(new_image)
x_test = x_test/255

# Invoke the interpreter
#time1 = time.time()
interpreter.set_tensor(input_details["index"], [x_test.astype(np.float32)])
interpreter.invoke()
y_pred1 = interpreter.get_tensor(output_details["index"])[0]

# save the predicted label
predicted_label1 = labels[y_pred1.argmax()]

time2 = time.time()
classification_time = np.round(time2-time1, 3)
print("Classification time =", classification_time, "seconds.")
print("Predicted label is:", predicted_label1) 

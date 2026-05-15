"""label_image for int8 tflite_micro_runtime (fixed safe version)."""

import argparse
import time
import numpy as np
from PIL import Image

import tflite_micro_runtime.interpreter as tflite


def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--image', default='test_cat.jpg')
    parser.add_argument('-m', '--model_file', default='cifar_10.tflite')
    parser.add_argument('-l', '--label_file', default='labels.txt')

    args = parser.parse_args()

    # -----------------------------
    # Load model
    # -----------------------------

    interpreter = tflite.Interpreter(model_path=args.model_file)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_index = input_details['index']
    output_index = output_details['index']

    input_dtype = input_details['dtype']
    output_dtype = output_details['dtype']

    _, height, width, channels = input_details['shape']

    # -----------------------------
    # MODEL VERIFY
    # -----------------------------

    print("\n========== MODEL INFO ==========")
    print("Model:", args.model_file)
    print("Input shape :", input_details['shape'])
    print("Output shape:", output_details['shape'])
    print("Input dtype :", input_dtype)
    print("Output dtype:", output_dtype)

    if len(input_details['shape']) != 4:
        raise RuntimeError("Expected 4D input tensor")

    if channels != 3:
        raise RuntimeError("Expected RGB input")

    if input_dtype != np.int8:
        raise RuntimeError(f"Expected int8 model, got {input_dtype}")

    print("Model verification PASSED")
    print("================================\n")

    # -----------------------------
    # Load image
    # -----------------------------

    img = Image.open(args.image).convert('RGB')
    img = img.resize((width, height), Image.Resampling.LANCZOS)

    img_np = np.asarray(img, dtype=np.uint8)

    print("========== IMAGE INFO ==========")
    print("Image:", args.image)
    print("Image shape:", img_np.shape)
    print("Image dtype:", img_np.dtype)
    print("Pixel range:", int(img_np.min()), "to", int(img_np.max()))
    print("================================\n")

    # -----------------------------
    # UINT8 -> INT8 conversion
    # -----------------------------

    img_int8 = (img_np.astype(np.int16) - 128).astype(np.int8)

    input_data = np.expand_dims(img_int8, axis=0)

    # -----------------------------
    # SET INPUT (SAFE FIX)
    # -----------------------------

    interpreter.set_tensor(input_index, input_data)

    # -----------------------------
    # INFERENCE
    # -----------------------------

    start_time = time.time()
    interpreter.invoke()
    elapsed_ms = (time.time() - start_time) * 1000.0

    # -----------------------------
    # OUTPUT
    # -----------------------------

    output_data = interpreter.get_tensor(output_index)
    results = np.squeeze(output_data)

    print("========== OUTPUT INFO ==========")
    print("Output shape:", output_data.shape)
    print("Output dtype:", output_data.dtype)
    print("Output min  :", results.min())
    print("Output max  :", results.max())
    print("=================================\n")

    labels = load_labels(args.label_file)

    top_k = results.argsort()[-3:][::-1]

    print("========== PREDICTIONS ==========")

    for i in top_k:

        # int8 output → simple scaling
        score = (float(results[i]) + 128.0) / 255.0

        print('{:08.6f}: {}'.format(score, labels[i]))

    print("=================================\n")
    print(f'Inference time: {elapsed_ms:.3f} ms')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf


class TFLiteInterpreter:
    def __init__(self, model_path, input_width, input_height, confidence_threshold=0.5, normalize_output=True, binarize_output=True):
        self.r = input_width
        self.c = input_height
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.confidence_threshold = confidence_threshold
        self.normalize_output = normalize_output
        self.binarize_output = binarize_output
    
    def predict(self, input_data):
        self.__set_input(input_data[None, ...])
        self.interpreter.invoke()
        output = self.__output_tensor()[0, ...]

        if self.normalize_output:
            output = self.__normalize_output(output)

        if self.binarize_output:
            output = self.__binarize_output(output)

        return output.squeeze().astype(np.uint8)

    def __input_tensor(self):
        tensor_index = self.interpreter.get_input_details()[0]["index"]
        return self.interpreter.tensor(tensor_index)()[0]

    def __output_tensor(self):
        # Compatibility for some models
        i = 0 if np.all(self.interpreter.get_output_details()[0]["shape"] == (1, self.c, self.r, 1)) else 1

        output_details = self.interpreter.get_output_details()[i]
        return self.interpreter.tensor(output_details["index"])()

    def __set_input(self, input_data):
        self.__input_tensor()[:, :] = input_data

    def __normalize_output(self, output):
        return (output - np.min(output)) / (np.max(output) - np.min(output))

    def __binarize_output(self, output):
        return output > self.confidence_threshold


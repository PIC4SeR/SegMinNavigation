import numpy as np
import tensorflow as tf

from row_crop_follow.utils.models import build_model_binary
from row_crop_follow.utils.mobilenet_v3 import MobileNetV3Large

gpus = tf.config.experimental.list_physical_devices("GPU")

if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


class TFGPUInterpreter:
    def __init__(self, model_file, input_width, input_height, confidence_threshold):
        backbone = MobileNetV3Large(
            input_shape=(input_width, input_height, 3),
            alpha=1.0,
            minimalistic=False,
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            classes=1,
            pooling="avg",
            dropout_rate=0.2,
            backend=tf.keras.backend,
            layers=tf.keras.layers,
            models=tf.keras.models,
            utils=tf.keras.utils,
        )
        self.threshold = confidence_threshold
        self.model = build_model_binary(backbone, 0.2, 1)
        self.model.load_weights(model_file)

    def predict(self, img):
        if len(img.shape) == 3:
            img = img[None]
        pred = self.model.predict(img)
        pred = pred > self.threshold
        return pred.squeeze().astype(np.uint8)

import numpy as np
import cv2

import matplotlib.pyplot as plt


def draw_histo(histo, vertical=True):
    histo_fig = np.zeros((len(histo), len(histo))).astype(np.uint8)  # histogram figure
    for c_index in range(len(histo)):
        for r_index in range(len(histo)):
            if r_index > len(histo) - histo[c_index]:
                histo_fig[r_index, c_index] = 255

    if not vertical:
        histo_fig = np.rot90(histo_fig, 3)

    return histo_fig


def show_image(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(1)


class SegController:
    def __init__(
        self,
        model_path,
        inference_device,
        confidence_threshold,
        depth_threshold,
        width,
        height,
        normalization,
        lin_vel_gain,
        ang_vel_gain,
        seg_min=False,
        weight_depth=False,
        show_result=False,
    ):
        self.buffer = None
        self.buffer_counter = 0
        self.width = width
        self.height = height
        self.lin_vel_gain = lin_vel_gain
        self.ang_vel_gain = ang_vel_gain
        self.seg_min = seg_min
        self.depth_threshold = depth_threshold
        self.weight_depth = weight_depth
        self.show_result = show_result

        if inference_device == "cpu":
            from row_crop_follow.tf_lite_interpreter import TFLiteInterpreter

            self.interpreter = TFLiteInterpreter(model_path, width, height, confidence_threshold=confidence_threshold)
        elif inference_device == "gpu":
            from row_crop_follow.tf_gpu_interpreter import TFGPUInterpreter

            self.interpreter = TFGPUInterpreter(model_path, width, height, confidence_threshold)
        else:
            raise ValueError("Invalid inference device. Choose 'cpu' or 'gpu'")

        self.normalization = normalization

        self._lin_vel = 0.0
        self._ang_vel = 0.0
        self._is_valid = False

    @property
    def lin_vel(self):
        return self._lin_vel

    @property
    def ang_vel(self):
        return self._ang_vel

    @property
    def is_valid(self):
        return self._is_valid

    def run(self, image, depth_image):
        self._is_valid = False

        image = cv2.resize(image.astype("uint8"), (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.show_result:
            show_image("Color", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        image = self.__normalize_image(image)
        segmentation_mask = self.interpreter.predict(image)
        if self.show_result:
            show_image("Segmentation", segmentation_mask * 255)

        if self.show_result:
            disp_depth_frame = np.where(segmentation_mask, depth_image, 0)
            disp_depth_frame = np.clip(disp_depth_frame, 0, self.depth_threshold)
            disp_depth_frame = (disp_depth_frame / np.max(disp_depth_frame) * 255).astype(np.uint8)
            show_image("Segmentation+Depth", disp_depth_frame)

        if self.buffer_counter == 0:
            self.buffer = np.copy(segmentation_mask)
            self.buffer_counter += 1
        elif self.buffer_counter < 3:
            # union
            self.buffer = segmentation_mask + self.buffer - segmentation_mask * self.buffer
            self.buffer_counter += 1
        else:
            self.buffer_counter = 0

            idexes_depth = self.__proccess_depth(depth_image)
            # cut the segmentation frame based on the depth information
            self.buffer[idexes_depth[:, 0], idexes_depth[:, 1]] = 0
            if self.seg_min and self.weight_depth:  # SegMinD
                self.buffer = self.__weight_inv_depth_frame(self.buffer, depth_image)
            histo = self.__get_histogram(self.buffer)
            self.__control_function(histo)

            if self.show_result:
                show_image("Histogram", draw_histo(histo))

        return segmentation_mask

    def __normalize_image(self, image):
        image = image / 255.0

        if self.normalization == "imagenet":
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            image = (image - mean) / std

        return image

    def __control_function(self, histo):
        if self.seg_min:  # SegMin
            command_x, ERROR_FLAG = self.__get_x_value_min(histo)
        else:  # SegZeros
            command_x, ERROR_FLAG = self.__get_x_value(histo)

        if not ERROR_FLAG:
            delta = command_x - (self.width / 2)
            K = delta**2 / (self.width / 2) ** 2
            self._ang_vel = -np.sign(delta) * self.ang_vel_gain * K
            self._lin_vel = self.lin_vel_gain * (1 - K)
            self._is_valid = True
        else:
            self._lin_vel = 0.0
            self._ang_vel = 0.0
            self._is_valid = False

    def __proccess_depth(self, depth_image):

        over_threshold_indexes = np.argwhere(depth_image > self.depth_threshold)
        zero_indexes = np.argwhere(depth_image == 0)
        return np.append(zero_indexes, over_threshold_indexes, axis=0)

    def __weight_inv_depth_frame(self, segmented_frame, depth_frame):
        """Used by SegMinD"""
        # Multipy element wise depth frame and segmented frame
        weighted_frame = np.multiply(segmented_frame, depth_frame)
        weighted_frame = (weighted_frame * 255 / (self.depth_threshold)).astype(np.uint8)
        return weighted_frame

    def __get_histogram(self, frame, smooth=True, window_length=11):
        histo = np.sum(frame, axis=0, dtype=np.uint8)

        if self.seg_min:
            histo = (histo / np.max(histo) * self.width).astype(np.uint8)

            if smooth:
                histo = np.convolve(np.ravel(histo), np.ones(window_length) / window_length, mode="same").astype(
                    np.uint8
                )
                histo[0 : window_length // 2 + 1] = (
                    np.ones(window_length // 2 + 1) * histo[window_length // 2 + 1]
                )  # padding on the left
                histo[-window_length // 2 :] = (
                    np.ones(window_length // 2 + 1) * histo[-window_length // 2 - 1]
                )  # padding on the right

        return histo

    def __get_x_value(self, histo):
        """USed by SegZeros"""
        command_x = int(self.width / 2)
        ERROR_FLAG = 0

        zero_ind = np.where(histo == 0)[0]  # index (x value) of the zeros
        cluster_list = []
        cluster_length = []
        last_index = 0  # last index of the cluster

        for index in range(len(zero_ind) - 1):
            # if the difference is greater than 1 then it is a different cluster
            if not (zero_ind[index + 1] - zero_ind[index]) == 1:
                # filtering from small clusters (noise)
                if len(zero_ind[last_index : index + 1]) > 3:
                    cluster_list.append(zero_ind[last_index : index + 1].tolist())
                    cluster_length.append(len(zero_ind[last_index : index + 1]))
                last_index = index + 1  #  to append the next cluster
        if len(zero_ind[last_index:]) > 3:
            cluster_list.append(zero_ind[last_index:].tolist())
            cluster_length.append(len(zero_ind[last_index:]))
        # The rover has reached the row end or it has not found free cluster
        if len(cluster_list) == 0:
            ERROR_FLAG = 1
            command_x = 0.0
            return command_x, ERROR_FLAG
        elif max(cluster_length) > int(0.8 * self.width):
            ERROR_FLAG = 1
            command_x = 0.0
            return command_x, ERROR_FLAG

        largest_cluster_index = np.argmax(np.array(cluster_length))
        zero_ind = cluster_list[largest_cluster_index]
        command_x = int((zero_ind[-1] - zero_ind[0]) / 2) + zero_ind[0]
        self.previous_command = command_x

        return command_x, ERROR_FLAG

    def __get_x_value_min(self, histo):
        """Used by SegMin and SegMinD"""
        command_x = int(self.width / 2)
        ERROR_FLAG = 0

        if np.sum(histo) == 0:
            ERROR_FLAG = 1
            command_x = 0.0
        else:
            min_arr = np.array(np.where(histo == histo.min()))  # finds where the minimum values are
            command_x = int(np.mean(min_arr))  # command depends on the center of the minimum values

        return command_x, ERROR_FLAG

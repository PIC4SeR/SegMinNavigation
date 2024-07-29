import os

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, SetParametersResult

from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

from row_crop_follow.tf_lite_interpreter import TFLiteInterpreter
from row_crop_follow.seg_controller import SegController


class SegControllerNode(Node):

    def __init__(self):

        super().__init__("segmentation_controller")

        ################################################################################################################
        ################################################################################################################
        # DECLARE AND GET PARAMETERS
        ################################################################################################################
        ################################################################################################################

        self.declare_parameters(
            namespace="",
            parameters=[
                (
                    "cmd_vel_topic",
                    "/cmd_vel",
                    ParameterDescriptor(
                        name="cmd_vel_topic",
                        description="Topic to publish the velocity commands",
                        type=ParameterType.PARAMETER_STRING,
                        read_only=True,
                    ),
                ),
                (
                    "rgb_topic",
                    "color/image_raw",
                    ParameterDescriptor(
                        name="rgb_topic",
                        description="Topic to subscribe the RGB image",
                        type=ParameterType.PARAMETER_STRING,
                        read_only=True,
                    ),
                ),
                (
                    "depth_topic",
                    "depth/image_raw",
                    ParameterDescriptor(
                        name="depth_topic",
                        description="Topic to subscribe the depth image",
                        type=ParameterType.PARAMETER_STRING,
                        read_only=True,
                    ),
                ),
                (
                    "model_file",
                    None,
                    ParameterDescriptor(
                        name="model_file",
                        description="Absolute path to the model file",
                        type=ParameterType.PARAMETER_STRING,
                        read_only=True,
                    ),
                ),
                (
                    "inference_device",
                    "cpu",
                    ParameterDescriptor(
                        name="inference_device",
                        description="Device to run the inference. Valid choices: cpu, gpu",
                        type=ParameterType.PARAMETER_STRING,
                        read_only=True,
                    ),
                ),
                (
                    "input_width_network",
                    224,
                    ParameterDescriptor(
                        name="input_width_network",
                        description="Input width of the network in pixels",
                        type=ParameterType.PARAMETER_INTEGER,
                        read_only=True,
                    ),
                ),
                (
                    "input_height_network",
                    224,
                    ParameterDescriptor(
                        name="input_height_network",
                        description="Input height of the network in pixels",
                        type=ParameterType.PARAMETER_INTEGER,
                        read_only=True,
                    ),
                ),
                (
                    "controller_type",
                    "SegZeros",
                    ParameterDescriptor(
                        name="controller_type",
                        description="Type of controller to use. Choiches: SegZeros, SegMin, SegMinD",
                        type=ParameterType.PARAMETER_STRING,
                        read_only=True,
                    ),
                ),
                (
                    "lin_vel_gain",
                    1.0,
                    ParameterDescriptor(
                        name="lin_vel_gain", description="Linear velocity gain", type=ParameterType.PARAMETER_DOUBLE
                    ),
                ),
                (
                    "ang_vel_gain",
                    3.0,
                    ParameterDescriptor(
                        name="ang_vel_gain", description="Angular velocity gain", type=ParameterType.PARAMETER_DOUBLE
                    ),
                ),
                (
                    "max_lin_vel",
                    0.5,
                    ParameterDescriptor(
                        name="max_lin_vel",
                        description="Maximum linear velocity [m/s]",
                        type=ParameterType.PARAMETER_DOUBLE,
                    ),
                ),
                (
                    "max_ang_vel",
                    0.5,
                    ParameterDescriptor(
                        name="max_ang_vel",
                        description="Maximum angular velocity [rad/s]",
                        type=ParameterType.PARAMETER_DOUBLE,
                    ),
                ),
                (
                    "ema_filter_alpha",
                    1.0,
                    ParameterDescriptor(
                        name="ema_filter_alpha",
                        description="Alpha parameter of the EMA filter. Put 1.0 if you don't want to use it. Range [0.0, 1.0]",
                        type=ParameterType.PARAMETER_DOUBLE,
                    ),
                ),
                (
                    "depth_threshold",
                    5.0,
                    ParameterDescriptor(
                        name="depth_threshold", description="Depth threshold [m]", type=ParameterType.PARAMETER_DOUBLE
                    ),
                ),
                (
                    "confidence_threshold",
                    0.7,
                    ParameterDescriptor(
                        name="confidence_threshold",
                        description="Confidence threshold for the segmentation model. Range [0.0, 1.0]",
                        type=ParameterType.PARAMETER_DOUBLE,
                    ),
                ),
                (
                    "show_result",
                    False,
                    ParameterDescriptor(
                        name="show_result",
                        description="Wheter the input and output images are shown or not",
                        type=ParameterType.PARAMETER_BOOL,
                    ),
                ),
                (
                    # TODO - Not implemented yet
                    "publish_processed_image",
                    True,
                    ParameterDescriptor(
                        name="publish_processed_image",
                        description="Wheter the processed image is published or not",
                        type=ParameterType.PARAMETER_BOOL,
                    ),
                ),
                (
                    "normalization",
                    "imagenet",
                    ParameterDescriptor(
                        name="normalization",
                        description="Normalization method. Valid choices: imagenet, none",
                        type=ParameterType.PARAMETER_STRING,
                        read_only=True,
                    ),
                ),
            ],
        )

        self.add_on_set_parameters_callback(self.cb_params)

        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").get_parameter_value().string_value
        self.rgb_topic = self.get_parameter("rgb_topic").get_parameter_value().string_value
        self.depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        self.model_file = self.get_parameter("model_file").get_parameter_value().string_value
        self.inference_device = self.get_parameter("inference_device").get_parameter_value().string_value
        self.input_width_network = self.get_parameter("input_width_network").get_parameter_value().integer_value
        self.input_height_network = self.get_parameter("input_height_network").get_parameter_value().integer_value
        self.controller_type = self.get_parameter("controller_type").get_parameter_value().string_value
        self.lin_vel_gain = self.get_parameter("lin_vel_gain").get_parameter_value().double_value
        self.ang_vel_gain = self.get_parameter("ang_vel_gain").get_parameter_value().double_value
        self.max_lin_vel = self.get_parameter("max_lin_vel").get_parameter_value().double_value
        self.max_ang_vel = self.get_parameter("max_ang_vel").get_parameter_value().double_value
        self.ema_filter_alpha = self.get_parameter("ema_filter_alpha").get_parameter_value().double_value
        self.depth_threshold = self.get_parameter("depth_threshold").get_parameter_value().double_value
        self.confidence_threshold = self.get_parameter("confidence_threshold").get_parameter_value().double_value
        self.show_result = self.get_parameter("show_result").get_parameter_value().bool_value
        self.publish_processed_image = self.get_parameter("publish_processed_image").get_parameter_value().bool_value
        self.normalization = self.get_parameter("normalization").get_parameter_value().string_value

        ################################################################################################################
        ################################################################################################################
        # END OF PARAMETERS SECTION
        ################################################################################################################
        ################################################################################################################

        self.controller = SegController(
            self.model_file,
            self.inference_device,
            self.confidence_threshold,
            self.depth_threshold,
            self.input_width_network,
            self.input_height_network,
            self.normalization,
            self.lin_vel_gain,
            self.ang_vel_gain,
            seg_min=("segmin" in self.controller_type.lower()),
            weight_depth=(self.controller_type.lower() == "segmind"),
            show_result=self.show_result,
        )

        self.get_logger().info(
            f"Started controller Seg{'MinD' if self.controller.seg_min and self.controller.weight_depth else 'Min' if self.controller.seg_min else 'Zeros'}"
        )

        self.lin_vel = 0.0
        self.ang_vel = 0.0

        self.depth_frame = None
        self.bridge = CvBridge()

        # # ROS 2 Publishers and Subscribers MUST BE LAST IN THE INITIALIZATION
        self.pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        # TODO - Implement the processed image publisher

        if "compressed" in self.rgb_topic:
            self.sub_rgb = self.create_subscription(CompressedImage, self.rgb_topic, self.image_callback, 10)
        else:
            self.sub_rgb = self.create_subscription(Image, self.rgb_topic, self.image_callback, 10)
        self.sub_depth = self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)

    # TODO - Implement the callback to update the parameters
    def cb_params(self, data: list[Parameter]):
        for parameter in data:
            self.get_logger().info(f"Requested parameter update: set {parameter.name} to {parameter.value}")
        return SetParametersResult(successful=True)

    def image_callback(self, data):
        if self.depth_frame is None:
            return

        #  Preprocess the image
        if isinstance(data, CompressedImage):
            bgr_frame = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
        elif isinstance(data, Image):
            bgr_frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        else:
            self.get_logger().error("Image type not supported")
            return

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        # Run the control loop
        self.controller.run(rgb_frame, self.depth_frame)

        if self.controller.is_valid:
            twist = Twist()
            self.lin_vel = self.ema_filter_alpha * self.controller.lin_vel + (1 - self.ema_filter_alpha) * self.lin_vel
            self.lin_vel = np.clip(self.lin_vel, -self.max_lin_vel, self.max_lin_vel)
            self.ang_vel = self.ema_filter_alpha * self.controller.ang_vel + (1 - self.ema_filter_alpha) * self.ang_vel
            self.ang_vel = np.clip(self.ang_vel, -self.max_ang_vel, self.max_ang_vel)
            twist.linear.x = float(self.lin_vel)
            twist.angular.z = float(self.ang_vel)
            self.pub.publish(twist)

    def depth_callback(self, data: Image):
        self.depth_frame = self.bridge.imgmsg_to_cv2(data, "passthrough")

        # If the format of the data is uint16 generally the depth is in millimeters. 
        # Convert to meters as for ROS 2  data format reccomendation.
        if self.depth_frame.dtype == "uint16":
            self.depth_frame = self.depth_frame.astype("float32") * 0.001

        self.depth_frame = cv2.resize(
            self.depth_frame,
            (self.input_width_network, self.input_height_network),
            interpolation=cv2.INTER_AREA,
        )


def main(args=None):
    rclpy.init(args=args)

    segmentation_controller = SegControllerNode()

    try:
        rclpy.spin(segmentation_controller)
    except KeyboardInterrupt:
        segmentation_controller.destroy_node()

    rclpy.try_shutdown()


if __name__ == "__main__":
    main()

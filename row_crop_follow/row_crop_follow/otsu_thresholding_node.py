import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, CompressedImage

from cv_bridge import CvBridge
from row_crop_follow.otsu_thresholding import OtsuThresholding

import cv2


class OtsuThresholdingNode(Node):

    def __init__(self):
        super().__init__("otsu_thresholding_node")

        propotional_criterion = self.declare_parameter("propotional_criterion", 1.5).get_parameter_value().double_value
        deviation_control_gain = (
            self.declare_parameter("deviation_control_gain", 0.01).get_parameter_value().double_value
        )
        correction_control_gain = (
            self.declare_parameter("correction_control_gain", -0.05).get_parameter_value().double_value
        )
        linear_velocity = self.declare_parameter("linear_velocity", 0.5).get_parameter_value().double_value
        grayscale_method = self.declare_parameter("grayscale_method", "cv2").get_parameter_value().string_value
        self.show_result = self.declare_parameter("show_result", False).get_parameter_value().bool_value
        if self.show_result:
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)

        self.controller = OtsuThresholding(
            proportional_criterion=propotional_criterion,
            deviation_control_gain=deviation_control_gain,
            correction_control_gain=correction_control_gain,
            linear_velocity=linear_velocity,
        )
        self.bridge = CvBridge()
        self.twist = Twist()

        cmd_vel_topic = self.declare_parameter("cmd_vel_topic", "cmd_vel").get_parameter_value().string_value
        self.cmd_vel_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.result_img_pub = self.create_publisher(Image, "result_img", 10)

        rgb_topic = self.declare_parameter("rgb_topic", "color/image_raw").get_parameter_value().string_value
        if "compressed" in rgb_topic:
            self.create_subscription(CompressedImage, rgb_topic, self.callback, 10)
        else:
            self.create_subscription(Image, rgb_topic, self.callback, 10)

    def callback(self, msg):
        if isinstance(msg, CompressedImage):
            img = self.bridge.compressed_imgmsg_to_cv2(msg)
        elif isinstance(msg, Image):
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        linear_velocity, angular_velocity, control, result_img = self.controller.run(img, self.show_result)

        if control == "NO LANE":
            self.get_logger().warn("Error in lane detection")

        self.twist.linear.x = linear_velocity
        self.twist.angular.z = angular_velocity
        self.cmd_vel_pub.publish(self.twist)
        self.result_img_pub.publish(self.bridge.cv2_to_imgmsg(result_img, "bgr8"))

    def on_shutdown(self):
        self.get_logger().info("Shutting down")
        self.controller.on_shutdown()


def main():
    rclpy.init()
    my_node = OtsuThresholdingNode()
    try:
        rclpy.spin(my_node)
    except KeyboardInterrupt:
        pass
    finally:
        my_node.destroy_node()  # cleans up pub-subs, etc
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

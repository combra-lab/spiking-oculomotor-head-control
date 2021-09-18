#!/usr/bin/env python

import rospy
import cv2
import math
import json
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError


class VisionNodeCenter:
    """ Vision node for robot head """

    def __init__(self):
        rospy.init_node("head_vision_node", anonymous=True)

        # Name of the camera
        self.camera_name = rospy.get_param('~camera_name')
        # Left or Right (0 for left, 1 for right)
        self.left_right = rospy.get_param('~left_right')
        # Red channel threshold
        self.red_threshold = rospy.get_param('~red_threshold')
        # Directory to receptive field file
        self.receptive_file_dir = rospy.get_param('~receptive_file_dir')
        # Publish frequency
        self.pub_freq = rospy.get_param('~pub_freq')
        # CV Bridge
        self.bridge = CvBridge()

        # Subscriber to image
        image_sub = rospy.Subscriber("/" + self.camera_name + "/image_raw", Image, self.image_cb)
        # Output publisher
        self.pub = rospy.Publisher("/" + self.camera_name + "/control_output", Float32MultiArray, queue_size=1)

        # Get receptive field mask and center
        self.rf_mask, self.rf_center = self.generate_receptive_field()

        # Init Image
        self.raw_image = None
        rospy.loginfo("Waiting for Image input")
        while self.raw_image is None and not rospy.is_shutdown():
            continue
        rospy.loginfo("Image input init")

    def generate_receptive_field(self):
        """
        Generate receptive field information

        Returns:
            rf_mask: receptive field mask
            rt_center: receptive field center index
            rf_weight: receptive field weight

        """
        with open(self.receptive_file_dir) as f:
            pixel_2_rf = json.load(f)

        # Generate rf mask
        rf_mask = np.zeros((720, 720))
        for xx in range(720):
            for yy in range(720):
                rf_idx = pixel_2_rf[str(xx) + ',' + str(yy)][0]
                rf_mask[xx, yy] = rf_idx

        # Generate rf center
        rf_num = int(np.max(rf_mask)) + 1
        rf_center = np.zeros((rf_num, 2))
        for rr in range(rf_num):
            pixels = np.argwhere(rf_mask == rr)
            np_pixels = np.array(pixels)
            rf_center[rr] = np.mean(np_pixels, axis=0)

        return rf_mask, rf_center

    def image_cb(self, msg):
        """
        Camera image callback function

        Args:
            msg (message): Image message

        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg)
        except CvBridgeError as e:
            print(e)

        # Take only red channel and 720x720 image
        resize_red_image = cv_image[:, 280:1000, 0]
        # Threshold image
        _, th_image = cv2.threshold(resize_red_image, self.red_threshold, 255, cv2.THRESH_BINARY)
        # Change to numpy array
        self.raw_image = np.array(th_image, dtype='uint8')

    def update_control_output(self, control_output, rf_idx):
        """
        Update control output base on receptive field index

        Args:
            control_output (list): control output
            rf_idx (int): receptive field index

        """
        rf_center_x, rf_center_y = self.rf_center[rf_idx, 0], self.rf_center[rf_idx, 1]
        max_w = math.log(1 + 360.0 / 288.0)

        # Vertical Movement
        if rf_center_x < 360:  # Up
            control_output[0] += math.log(1 + (360.0 - rf_center_x)/288.0) / max_w
        elif rf_center_x > 360:  # Down
            control_output[1] += math.log(1 + (rf_center_x - 360.0)/288.0) / max_w

        # Horizontal Movement
        if rf_center_y < 360:  # Left
            control_output[2] += math.log(1 + (360.0 - rf_center_y)/288.0) / max_w
            # Addition dimension output for left right eye not added here
            # Detail see original code midbrain/new_brain.h/computeColliculusInput
        elif rf_center_y > 360:  # Right
            control_output[3] += math.log(1 + (rf_center_y - 360.0)/288.0) / max_w
            # Addition dimension output for left right eye not added here
            # Detail see original code midbrain/new_brain.h/computeColliculusInput

    def run_node(self):
        """
        Run Camera Node

        """
        ros_rate = rospy.Rate(self.pub_freq)

        while not rospy.is_shutdown():
            laser_point_image = self.raw_image.copy()
            # Get pixel index for center of laser point pixel
            laser_point_pixel = np.argwhere(laser_point_image == 255)
            rf_image = np.zeros((720, 720), dtype='uint8')
            control_output = [0, 0, 0, 0]  # Up, Down, Left, Right

            if laser_point_pixel.shape[0] > 0:
                laser_point_center = laser_point_pixel.mean(axis=0).astype(np.int)
            
                rf_idx = int(self.rf_mask[laser_point_center[0], laser_point_center[1]])
                self.update_control_output(control_output, rf_idx)
            
                rf_image[self.rf_mask == rf_idx] = 255

            show_image = cv2.hconcat((laser_point_image, rf_image))
            cv2.imshow(self.camera_name, show_image)
            cv2.waitKey(3)

            # Publish control output
            pub_msg = Float32MultiArray()
            pub_msg.data = control_output
            self.pub.publish(pub_msg)

            ros_rate.sleep()


if __name__ == '__main__':
    node = VisionNodeCenter()
    node.run_node()

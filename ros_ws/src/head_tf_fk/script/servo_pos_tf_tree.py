#!/usr/bin/env python

import rospy
import tf_conversions
import tf2_ros
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState
import numpy as np
import matplotlib.pyplot as plt
import math


class HeadExperimentTFTree:
    """ Class for building head experiment tf tree """

    def __init__(self,
                 base_link_2_wall_y=-(0.3 + 0.191 * 1.5),
                 base_link_2_laser_base_y=-0.191,
                 laser_base_2_laser_pan_z=0.086,
                 laser_pan_2_laser_tilt_z=0.043,
                 laser_tilt_2_laser_z=0.035,
                 # neck_tilt_2_eye_pan_x=0.055,
                 neck_tilt_2_eye_pan_x=0.060,
                 neck_tilt_2_eye_pan_y=0.005,
                 neck_tilt_2_eye_pan_z=0.068,
                 eye_pan_2_eye_tilt_z=0.044,
                 eye_tilt_2_camera_z=0.054,
                 scale_factor=10.0,
                 ros_rate=20):
        """

        Args:
            base_link_2_wall_y (float): base_link to wall on y axis
            base_link_2_laser_base_y (float): base_link to laser_base on y axis
            laser_base_2_laser_pan_z (float): laser_base to laser_pan on z axis
            laser_pan_2_laser_tilt_z (float): laser_pan to laser_tilt on z axis
            laser_tilt_2_laser_z (float): laser_tilt to laser on z axis
            neck_tilt_2_eye_pan_x (float): neck_tilt to eye_pan on x axis
            neck_tilt_2_eye_pan_y (float): neck_tilt to eye_pan on y axis
            neck_tilt_2_eye_pan_z (float): neck_tilt to eye_pan on z axis
            eye_pan_2_eye_tilt_z (float): eye_pan to eye_tilt on z axis
            eye_tilt_2_camera_z (float): eye_tilt to camera on z axis
            scale_factor (float): scale factor for distance in meter
            ros_rate (int): ROS rate
        """
        self.base_link_2_wall_y = base_link_2_wall_y * scale_factor

        # Laser pointer configuration
        self.base_link_2_laser_base_y = base_link_2_laser_base_y * scale_factor
        self.laser_base_2_laser_pan_z = laser_base_2_laser_pan_z * scale_factor
        self.laser_pan_2_laser_tilt_z = laser_pan_2_laser_tilt_z * scale_factor
        self.laser_tilt_2_laser_z = laser_tilt_2_laser_z * scale_factor

        # Head configuration
        self.neck_tilt_2_eye_pan_x = neck_tilt_2_eye_pan_x * scale_factor
        self.neck_tilt_2_eye_pan_y = neck_tilt_2_eye_pan_y * scale_factor
        self.neck_tilt_2_eye_pan_z = neck_tilt_2_eye_pan_z * scale_factor
        self.eye_pan_2_eye_tilt_z = eye_pan_2_eye_tilt_z * scale_factor
        self.eye_tilt_2_camera_z = eye_tilt_2_camera_z * scale_factor

        rospy.init_node("head_exp_tf_br")
        self.tf_br = tf2_ros.TransformBroadcaster()
        self.ros_rate = rospy.Rate(ros_rate)

        # create laser pointer joint state subscriber
        self.laser_joint_state = None
        self.laser_sub = rospy.Subscriber('/laser/joint_states', JointState, self.laser_js_cb, queue_size=5)

        # create head joint state subscriber
        self.head_joint_state = None
        self.head_sub = rospy.Subscriber('/full_head/joint_states', JointState, self.head_js_cb, queue_size=5)

        while ((self.laser_joint_state is None) or (self.head_joint_state is None)) and (not rospy.is_shutdown()):
            continue

        rospy.loginfo("Laser Pointer and Head Joint State Init ...")

    @staticmethod
    def generate_tf_msg(trans_mat, parent_frame_id, frame_id):
        position = [trans_mat[0, 3], trans_mat[1, 3], trans_mat[2, 3]]
        quat = tf_conversions.transformations.quaternion_from_matrix(trans_mat)

        tf_msg = TransformStamped()

        tf_msg.header.stamp = rospy.Time.now()
        tf_msg.header.frame_id = parent_frame_id
        tf_msg.child_frame_id = frame_id
        tf_msg.transform.translation.x = position[0]
        tf_msg.transform.translation.y = position[1]
        tf_msg.transform.translation.z = position[2]
        tf_msg.transform.rotation.x = quat[0]
        tf_msg.transform.rotation.y = quat[1]
        tf_msg.transform.rotation.z = quat[2]
        tf_msg.transform.rotation.w = quat[3]

        return tf_msg

    @staticmethod
    def compute_eye_wall_project_y(neck_pan, neck_tilt, eye_pan, eye_tilt,
                                   pan_x, pan_y, pan_z, tilt_z, camera_z, wall_dis):
        """
        Compute project y axis distance from eye to wall

        Args:
            neck_pan (float): rad of neck pan servo
            neck_tilt (float): rad of neck tilt servo
            eye_pan (float): rad of eye pan servo
            eye_tilt (float): rad of eye tilt servo
            pan_x (float): neck_tilt to eye_pan x axis
            pan_y (float): neck_tilt to eye_pan y axis
            pan_z (float): neck_tilt to eye_pan z axis
            tilt_z (float): eye_pan to eye_tilt z axis
            camera_z (float): eye_tilt to camera z axis
            wall_dis (float): wall distance to base_link

        Returns:
            eye_wall_y: eye to wall y axis

        """
        up_a = -camera_z * (math.sin(eye_tilt) * (
                math.sin(neck_pan) * math.sin(eye_pan) - math.cos(neck_pan) * math.cos(neck_tilt) * math.cos(
            eye_pan)) - math.cos(neck_pan) * math.sin(neck_tilt) * math.cos(eye_tilt))
        up_b = tilt_z * math.cos(neck_pan) * math.sin(neck_tilt) - pan_x * math.sin(neck_pan) - pan_y * math.cos(
            neck_pan) * math.cos(neck_tilt) + pan_z * math.cos(neck_pan) * math.sin(neck_tilt)
        down = -math.cos(eye_tilt) * (
                math.sin(neck_pan) * math.sin(eye_pan) - math.cos(neck_pan) * math.cos(neck_tilt) * math.cos(
            eye_pan)) - math.cos(neck_pan) * math.sin(neck_tilt) * math.sin(eye_tilt)
        eye_wall_y = (wall_dis + up_a + up_b) / down
        return eye_wall_y

    def laser_js_cb(self, msg):
        """
        Laser pointer joint state callback

        Args:
            msg (JointState): Message

        """
        self.laser_joint_state = [msg.position[msg.name.index("laser_pan")],
                                  msg.position[msg.name.index("laser_tilt")]]

    def head_js_cb(self, msg):
        """
        Head joint state callback

        Args:
            msg (JointState): Message

        """
        self.head_joint_state = [msg.position[msg.name.index("neck_pan")],
                                 msg.position[msg.name.index("neck_tilt1")],
                                 msg.position[msg.name.index("left_eye_pan")],
                                 msg.position[msg.name.index("left_eye_tilt")],
                                 msg.position[msg.name.index("right_eye_pan")],
                                 msg.position[msg.name.index("right_eye_tilt")]]

    def update_laser_pointer_tf(self, laser_pan, laser_tilt):
        """
        Update laser pointer tf tree for one step

        Args:
            laser_pan (float): rad of laser pan servo
            laser_tilt (float): rad of laser tilt servo

        Returns:
            laser_project_pos: laser position project to wall

        """
        # Compute transformation matrix for each link
        base_link_2_laser_pan_trans = np.array([[math.cos(laser_pan), -math.sin(laser_pan), 0., 0.],
                                                [math.sin(laser_pan), math.cos(laser_pan), 0.,
                                                 self.base_link_2_laser_base_y],
                                                [0., 0., 1., self.laser_base_2_laser_pan_z],
                                                [0., 0., 0., 1.]])

        laser_pan_2_laser_tilt_trans = np.array([[1., 0., 0., 0.],
                                                 [0., math.cos(laser_tilt), -math.sin(laser_tilt), 0.],
                                                 [0., math.sin(laser_tilt), math.cos(laser_tilt),
                                                  self.laser_pan_2_laser_tilt_z],
                                                 [0., 0., 0., 1.]])

        laser_tilt_2_laser_trans = np.array([[1., 0., 0., 0.],
                                             [0., 1., 0., 0.],
                                             [0., 0., 1., self.laser_tilt_2_laser_z],
                                             [0., 0., 0., 1.]])

        laser_2_wall_y = (self.base_link_2_wall_y - self.base_link_2_laser_base_y) / (
                math.cos(laser_pan) * math.cos(laser_tilt)) + (
                                 math.sin(laser_tilt) * self.laser_tilt_2_laser_z) / math.cos(laser_tilt)
        laser_2_wall_trans = np.array([[1., 0., 0., 0.],
                                       [0., 1., 0., laser_2_wall_y],
                                       [0., 0., 1., 0.],
                                       [0., 0., 0., 1.]])

        # Compute final transformation from base link to laser point on the wall
        base_link_2_wall_laser_point_trans = np.matmul(np.matmul(np.matmul(base_link_2_laser_pan_trans,
                                                                           laser_pan_2_laser_tilt_trans),
                                                                 laser_tilt_2_laser_trans),
                                                       laser_2_wall_trans)
        wall_laser_point = base_link_2_wall_laser_point_trans[:3, 3]

        # Broadcast TF tree
        tf_laser_pan = HeadExperimentTFTree.generate_tf_msg(base_link_2_laser_pan_trans, "base_link", "laser_pan")
        tf_laser_tilt = HeadExperimentTFTree.generate_tf_msg(laser_pan_2_laser_tilt_trans, "laser_pan", "laser_tilt")
        tf_laser = HeadExperimentTFTree.generate_tf_msg(laser_tilt_2_laser_trans, "laser_tilt", "laser")
        tf_wall_laser = HeadExperimentTFTree.generate_tf_msg(laser_2_wall_trans, "laser", "wall_laser")

        self.tf_br.sendTransform(tf_laser_pan)
        self.tf_br.sendTransform(tf_laser_tilt)
        self.tf_br.sendTransform(tf_laser)
        self.tf_br.sendTransform(tf_wall_laser)

        return wall_laser_point

    def update_head_tf(self, neck_pan, neck_tilt, left_eye_pan, left_eye_tilt, right_eye_pan, right_eye_tilt):
        """
        Update head tf tree for one step

        Args:
            neck_pan (float): rad of neck pan servo
            neck_tilt (float): rad of neck tilt servo
            left_eye_pan (float): rad for left eye pan servo
            left_eye_tilt (float): rad for left eye tilt servo
            right_eye_pan (float): rad for right eye pan servo
            right_eye_tilt (float): rad for right eye tilt servo

        Returns:
            left_project_pos: left eye position project to wall
            right_project_pos: right eye position project to wall

        """
        # print(neck_tilt)
        # if abs(neck_pan) >= 1 or abs(neck_tilt) > 1 or abs(left_eye_pan) > 1 or abs(left_eye_tilt) > 1 or abs(right_eye_pan) > 1 or abs(right_eye_tilt) > 1:
        #     print(neck_pan, neck_tilt, left_eye_pan, left_eye_tilt, right_eye_pan, right_eye_tilt)
        # Compute transformation matrix for neck
        base_link_2_neck_pan_trans = np.array([[math.cos(neck_pan), -math.sin(neck_pan), 0., 0.],
                                               [math.sin(neck_pan), math.cos(neck_pan), 0., 0.],
                                               [0., 0., 1., self.laser_base_2_laser_pan_z],
                                               [0., 0., 0., 1.]])

        neck_pan_2_neck_tilt_trans = np.array([[1., 0., 0., 0.],
                                               [0., math.cos(neck_tilt), -math.sin(neck_tilt), 0.],
                                               [0., math.sin(neck_tilt), math.cos(neck_tilt),
                                                self.laser_pan_2_laser_tilt_z],
                                               [0., 0., 0., 1.]])

        # Compute transformation matrix for left eye
        neck_tilt_2_left_pan_trans = np.array([[math.cos(left_eye_pan), -math.sin(left_eye_pan), 0.,
                                                self.neck_tilt_2_eye_pan_x],
                                               [math.sin(left_eye_pan), math.cos(left_eye_pan), 0.,
                                                self.neck_tilt_2_eye_pan_y],
                                               [0., 0., 1., self.neck_tilt_2_eye_pan_z],
                                               [0., 0., 0., 1.]])

        left_pan_2_left_tilt_trans = np.array([[1., 0., 0., 0.],
                                               [0., math.cos(left_eye_tilt), -math.sin(left_eye_tilt), 0.],
                                               [0., math.sin(left_eye_tilt), math.cos(left_eye_tilt),
                                                self.eye_pan_2_eye_tilt_z],
                                               [0., 0., 0., 1.]])
        left_tilt_2_left_camera_trans = np.array([[1., 0., 0., 0.],
                                                  [0., 1., 0., 0.],
                                                  [0., 0., 1., self.eye_tilt_2_camera_z],
                                                  [0., 0., 0., 1.]])

        left_camera_2_wall_y = HeadExperimentTFTree.compute_eye_wall_project_y(neck_pan, neck_tilt,
                                                                               left_eye_pan, left_eye_tilt,
                                                                               self.neck_tilt_2_eye_pan_x,
                                                                               self.neck_tilt_2_eye_pan_y,
                                                                               self.neck_tilt_2_eye_pan_z,
                                                                               self.eye_pan_2_eye_tilt_z,
                                                                               self.eye_tilt_2_camera_z,
                                                                               self.base_link_2_wall_y)
        left_camera_2_wall_trans = np.array([[1., 0., 0., 0.],
                                             [0., 1., 0., left_camera_2_wall_y],
                                             [0., 0., 1., 0.],
                                             [0., 0., 0., 1.]])

        # Compute transformation matrix for right eye
        neck_tilt_2_right_pan_trans = np.array([[math.cos(right_eye_pan), -math.sin(right_eye_pan), 0.,
                                                 -self.neck_tilt_2_eye_pan_x],
                                                [math.sin(right_eye_pan), math.cos(right_eye_pan), 0.,
                                                 self.neck_tilt_2_eye_pan_y],
                                                [0., 0., 1., self.neck_tilt_2_eye_pan_z],
                                                [0., 0., 0., 1.]])

        right_pan_2_right_tilt_trans = np.array([[1., 0., 0., 0.],
                                                 [0., math.cos(right_eye_tilt), -math.sin(right_eye_tilt), 0.],
                                                 [0., math.sin(right_eye_tilt), math.cos(right_eye_tilt),
                                                  self.eye_pan_2_eye_tilt_z],
                                                 [0., 0., 0., 1.]])
        right_tilt_2_right_camera_trans = np.array([[1., 0., 0., 0.],
                                                    [0., 1., 0., 0.],
                                                    [0., 0., 1., self.eye_tilt_2_camera_z],
                                                    [0., 0., 0., 1.]])

        right_camera_2_wall_y = HeadExperimentTFTree.compute_eye_wall_project_y(neck_pan, neck_tilt,
                                                                                right_eye_pan, right_eye_tilt,
                                                                                -self.neck_tilt_2_eye_pan_x,
                                                                                self.neck_tilt_2_eye_pan_y,
                                                                                self.neck_tilt_2_eye_pan_z,
                                                                                self.eye_pan_2_eye_tilt_z,
                                                                                self.eye_tilt_2_camera_z,
                                                                                self.base_link_2_wall_y)
        right_camera_2_wall_trans = np.array([[1., 0., 0., 0.],
                                              [0., 1., 0., right_camera_2_wall_y],
                                              [0., 0., 1., 0.],
                                              [0., 0., 0., 1.]])

        # Compute final transformation from base link to left and right eyes on the wall
        base_link_2_left_wall_trans = np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(base_link_2_neck_pan_trans,
                                                                                        neck_pan_2_neck_tilt_trans),
                                                                              neck_tilt_2_left_pan_trans),
                                                                    left_pan_2_left_tilt_trans),
                                                          left_tilt_2_left_camera_trans),
                                                left_camera_2_wall_trans)
        base_link_2_right_wall_trans = np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(base_link_2_neck_pan_trans,
                                                                                         neck_pan_2_neck_tilt_trans),
                                                                               neck_tilt_2_right_pan_trans),
                                                                     right_pan_2_right_tilt_trans),
                                                           right_tilt_2_right_camera_trans),
                                                 right_camera_2_wall_trans)
        left_project_pos = base_link_2_left_wall_trans[:3, 3]
        right_project_pos = base_link_2_right_wall_trans[:3, 3]

        # Broadcast TF tree
        tf_neck_pan = HeadExperimentTFTree.generate_tf_msg(base_link_2_neck_pan_trans, "base_link", "neck_pan")
        tf_neck_tilt = HeadExperimentTFTree.generate_tf_msg(neck_pan_2_neck_tilt_trans, "neck_pan", "neck_tilt")
        tf_left_pan = HeadExperimentTFTree.generate_tf_msg(neck_tilt_2_left_pan_trans, "neck_tilt", "left_pan")
        tf_left_tilt = HeadExperimentTFTree.generate_tf_msg(left_pan_2_left_tilt_trans, "left_pan", "left_tilt")
        tf_left_camera = HeadExperimentTFTree.generate_tf_msg(left_tilt_2_left_camera_trans, "left_tilt", "left_camera")
        tf_left_wall = HeadExperimentTFTree.generate_tf_msg(left_camera_2_wall_trans, "left_camera", "left_point")
        tf_right_pan = HeadExperimentTFTree.generate_tf_msg(neck_tilt_2_right_pan_trans, "neck_tilt", "right_pan")
        tf_right_tilt = HeadExperimentTFTree.generate_tf_msg(right_pan_2_right_tilt_trans, "right_pan", "right_tilt")
        tf_right_camera = HeadExperimentTFTree.generate_tf_msg(right_tilt_2_right_camera_trans, "right_tilt",
                                                               "right_camera")
        tf_right_wall = HeadExperimentTFTree.generate_tf_msg(right_camera_2_wall_trans, "right_camera", "right_point")

        self.tf_br.sendTransform(tf_neck_pan)
        self.tf_br.sendTransform(tf_neck_tilt)
        self.tf_br.sendTransform(tf_left_pan)
        self.tf_br.sendTransform(tf_left_tilt)
        self.tf_br.sendTransform(tf_left_camera)
        self.tf_br.sendTransform(tf_left_wall)
        self.tf_br.sendTransform(tf_right_pan)
        self.tf_br.sendTransform(tf_right_tilt)
        self.tf_br.sendTransform(tf_right_camera)
        self.tf_br.sendTransform(tf_right_wall)

        return left_project_pos, right_project_pos

    def run(self, max_ros_ita):
        """
        Run ROS node

        Returns:
            wall_laser_point_list: list of wall laser point
            left_point_list: list of left point
            right_point_list: list of right point

        """
        wall_laser_point_list = np.zeros((max_ros_ita, 3))
        left_point_list = np.zeros((max_ros_ita, 3))
        right_point_list = np.zeros((max_ros_ita, 3))

        ros_ita = 0
        while not rospy.is_shutdown():
            wall_laser_point = self.update_laser_pointer_tf(self.laser_joint_state[0], self.laser_joint_state[1])
            left_point, right_point = self.update_head_tf(self.head_joint_state[0], -self.head_joint_state[1],
                                                          self.head_joint_state[2], -self.head_joint_state[3],
                                                          self.head_joint_state[4], -self.head_joint_state[5])
            wall_laser_point_list[ros_ita] = wall_laser_point[:]
            left_point_list[ros_ita] = left_point[:]
            right_point_list[ros_ita] = right_point[:]
            ros_ita += 1
            if ros_ita == max_ros_ita:
                break
            self.ros_rate.sleep()

        return wall_laser_point_list, left_point_list, right_point_list


if __name__ == '__main__':
    ros_ita_num = 20 * 50
    ros_node = HeadExperimentTFTree()
    laser_list, left_list, right_list = ros_node.run(ros_ita_num)

    import pickle
    import os
    with open(os.getcwd() + '/exp_results/loihi_step_exp.p', 'wb+') as fw:
        pickle.dump([laser_list, left_list, right_list], fw)

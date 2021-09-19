import rospy
from arbotix_msgs.srv import SetSpeed
from std_msgs.msg import Float64, Float32MultiArray
import numpy as np
import math
import copy
import time


class HeadRosControl:
    """ Control Head using ROS and SNN on Loihi """

    def __init__(self, eye_pan_increase, eye_pan_limit, eye_tilt_increase, eye_tile_limit,
                 neck_pan_increase, neck_pan_limit, neck_tilt_increase, neck_tilt_limit,
                 input_amp, move_speed, ros_rate):
        """

        Args:
            eye_pan_increase (float): eye pan increase rate
            eye_pan_limit (float): eye pan position limit
            eye_tilt_increase (float): eye tilt increase rate
            eye_tile_limit (float): eye tilt position limit
            neck_pan_increase (float): neck pan increase rate
            neck_pan_limit (float): neck pan position limit
            neck_tilt_increase (float): neck tilt increase rate
            neck_tilt_limit (float): neck tilt position limit
            move_speed (float): move speed for servo between positions
            input_amp (float): input amplifier for control output
            ros_rate (int): ros node update rate
        """
        rospy.init_node("head_ros_control")
        self.eye_pan_increase = eye_pan_increase
        self.eye_pan_limit = eye_pan_limit
        self.eye_tilt_increase = eye_tilt_increase
        self.eye_tilt_limit = eye_tile_limit
        self.neck_pan_increase = neck_pan_increase
        self.neck_pan_limit = neck_pan_limit
        self.neck_tilt_increase = neck_tilt_increase
        self.neck_tilt_limit = neck_tilt_limit
        self.input_amp = input_amp
        self.move_speed = move_speed
        self.ros_rate = rospy.Rate(ros_rate)

        # Create eye output control subscriber
        self.left_eye_output = None
        self.right_eye_output = None
        self.left_eye_sub = rospy.Subscriber('/left_cam/control_output', Float32MultiArray,
                                             self.left_eye_cb, queue_size=1)
        self.right_eye_sub = rospy.Subscriber('/right_cam/control_output', Float32MultiArray,
                                              self.right_eye_cb, queue_size=1)
        rospy.loginfo("Wait Left and Right Eye Init ...")
        while self.left_eye_output is None and self.right_eye_output is None and not rospy.is_shutdown():
            continue
        rospy.loginfo("Left and Right Eye Init finished ...")

        # Create Service and Publisher for head control
        self.joint_position = np.zeros(7)
        self.joint_name_list = ['neck_pan', 'neck_tilt1', 'neck_tilt2',
                                'left_eye_pan', 'left_eye_tilt',
                                'right_eye_pan', 'right_eye_tilt']
        self.speed_srv_list = []
        self.pos_pub_list = []
        for name in self.joint_name_list:
            speed_srv_name = '/full_head/' + name + '/set_speed'
            self.speed_srv_list.append(rospy.ServiceProxy(speed_srv_name, SetSpeed))
            pos_pub_name = '/full_head/' + name + '/command'
            self.pos_pub_list.append(rospy.Publisher(pos_pub_name, Float64, queue_size=5))

        # Set all servo with moving speed
        for speed_srv in self.speed_srv_list[:3]:
            try:
                speed_srv(move_speed / 4.)
            except rospy.ServiceException as e:
                print("Setting move speed Failed: %s" % e)
        for speed_srv in self.speed_srv_list[3:]:
            try:
                speed_srv(move_speed)
            except rospy.ServiceException as e:
                print("Setting move speed Failed: %s" % e)

    def left_eye_cb(self, msg):
        """
        Callback function for left eye

        Args:
            msg (Message): Message

        """
        self.left_eye_output = msg.data

    def right_eye_cb(self, msg):
        """
        Callback function for right eye

        Args:
            msg (Message): Message

        """
        self.right_eye_output = msg.data

    def run_node(self, max_ros_ita, encoder_channel, decoder_channel):
        """
        Run ROS node for the head

        Args:
            max_ros_ita (int): max ros iteration
            encoder_channel (Loihi Channel): Encoder input channel
            decoder_channel (Loihi Channel): Decoder input channel

        """
        ros_ita = 0
        
        while not rospy.is_shutdown():
            
            left_eye_current = copy.deepcopy(self.left_eye_output)
            right_eye_current = copy.deepcopy(self.right_eye_output)
            print("Left raw current: ", left_eye_current, " Right raw current: ", right_eye_current)
            left_eye_current = [int(current * self.input_amp) for current in left_eye_current]
            right_eye_current = [int(current * self.input_amp) for current in right_eye_current]

            # Loihi SNN computation
            mutual_up = max(max(left_eye_current[0] - left_eye_current[1],
                                right_eye_current[0] - right_eye_current[1]), 0)
            mutual_down = max(max(left_eye_current[1] - left_eye_current[0],
                                  right_eye_current[1] - right_eye_current[0]), 0)
            left_eye_left = max(left_eye_current[2] - left_eye_current[3], 0)
            left_eye_right = max(left_eye_current[3] - left_eye_current[2], 0)
            right_eye_left = max(right_eye_current[2] - right_eye_current[3], 0)
            right_eye_right = max(right_eye_current[3] - right_eye_current[2], 0)
            encoder_channel.write(6, [mutual_up, mutual_down, left_eye_left, left_eye_right,
                                      right_eye_left, right_eye_right])
            delta_motor_spikes = decoder_channel.read(10)
            print("Control output: ", delta_motor_spikes)

            # Update neck positions
            self.joint_position[0] += self.neck_pan_increase * (delta_motor_spikes[8] - delta_motor_spikes[9])
            self.joint_position[1] += self.neck_tilt_increase * (delta_motor_spikes[6] - delta_motor_spikes[7])
            self.joint_position[2] = self.joint_position[1]
            if abs(self.joint_position[0]) > self.neck_pan_limit:
                self.joint_position[0] = math.copysign(1.0, self.joint_position[0]) * self.neck_pan_limit
            if abs(self.joint_position[1]) > self.neck_tilt_limit:
                self.joint_position[1] = math.copysign(1.0, self.joint_position[1]) * self.neck_tilt_limit
                self.joint_position[2] = self.joint_position[1]

            # Update left eye positions
            self.joint_position[3] += self.eye_pan_increase * (delta_motor_spikes[2] - delta_motor_spikes[3])
            self.joint_position[4] += self.eye_tilt_increase * (delta_motor_spikes[0] - delta_motor_spikes[1])
            if abs(self.joint_position[3]) > self.eye_pan_limit:
                self.joint_position[3] = math.copysign(1.0, self.joint_position[3]) * self.eye_pan_limit
            if abs(self.joint_position[4]) > self.eye_tilt_limit:
                self.joint_position[4] = math.copysign(1.0, self.joint_position[4]) * self.eye_tilt_limit

            # Update right eye positions
            self.joint_position[5] += self.eye_pan_increase * (delta_motor_spikes[4] - delta_motor_spikes[5])
            self.joint_position[6] += self.eye_tilt_increase * (delta_motor_spikes[0] - delta_motor_spikes[1])
            if abs(self.joint_position[5]) > self.eye_pan_limit:
                self.joint_position[5] = math.copysign(1.0, self.joint_position[5]) * self.eye_pan_limit
            if abs(self.joint_position[6]) > self.eye_tilt_limit:
                self.joint_position[6] = math.copysign(1.0, self.joint_position[6]) * self.eye_tilt_limit

            # Control servos
            for dd, pos_pub in enumerate(self.pos_pub_list, 0):
                pos_msg = Float64()
                pos_msg.data = self.joint_position[dd]
                pos_pub.publish(pos_msg)

            ros_ita += 1
            if ros_ita == max_ros_ita:
                break
            self.ros_rate.sleep()


if __name__ == '__main__':
    from snn_loihi.setup_loihi_snn import setup_full_head_snn, compile_single_joint_head_snn
    from params import cfg

    eye_pan_inc = cfg['head']['eye_pan_inc']
    eye_pan_lim = cfg['head']['eye_pan_lim']
    eye_tilt_inc = cfg['head']['eye_tilt_inc']
    eye_tilt_lim = cfg['head']['eye_tilt_lim']
    neck_pan_inc = cfg['head']['neck_pan_inc']
    neck_pan_lim = cfg['head']['neck_pan_lim']
    neck_tilt_inc = cfg['head']['neck_tilt_inc']
    neck_tilt_lim = cfg['head']['neck_tilt_lim']
    move_spd = cfg['head']['servo_move_spd']
    in_amp = cfg['head']['input_amp']
    rate = cfg['head']['ros_rate']
    control_node = HeadRosControl(eye_pan_inc, eye_pan_lim, eye_tilt_inc, eye_tilt_lim,
                                  neck_pan_inc, neck_pan_lim, neck_tilt_inc, neck_tilt_lim,
                                  in_amp, move_spd, rate)
    max_ita = rate * cfg['head']['run_time']
    loihi_ts = max_ita * 100
    loihi_net, in_conn_dict = setup_full_head_snn()
    loihi_board, in_channel, out_channel = compile_single_joint_head_snn(loihi_net, in_conn_dict)
    loihi_board.startDriver()
    loihi_board.run(loihi_ts, aSync=True)
    control_node.run_node(max_ita, in_channel, out_channel)
    loihi_board.finishRun()
    loihi_board.disconnect()


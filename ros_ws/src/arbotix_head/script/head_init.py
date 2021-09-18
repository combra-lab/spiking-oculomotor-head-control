#!/usr/bin/env python

import rospy
from arbotix_msgs.srv import SetSpeed
from std_msgs.msg import Float64


def head_init():
    """
    Move head to init position
    """

    rospy.init_node("init_head_pos")
    speed_srv_list = []
    pos_pub_list = []
    name_list = ['neck_pan', 'neck_tilt1', 'neck_tilt2',
                 'left_eye_pan', 'left_eye_tilt',
                 'right_eye_pan', 'right_eye_tilt']
    for name in name_list:
        speed_srv_name = '/full_head/' + name + '/set_speed'
        speed_srv_list.append(rospy.ServiceProxy(speed_srv_name, SetSpeed))
        pos_pub_name = '/full_head/' + name + '/command'
        pos_pub_list.append(rospy.Publisher(pos_pub_name, Float64, queue_size=5))

    for speed_srv in speed_srv_list:
        try:
            speed_srv(2.0)
        except rospy.ServiceException as e:
            print("Set Speed Failed ...")

    for pos_pub in pos_pub_list:
        pos_msg = Float64()
        pos_msg.data = 0.0
        pos_pub.publish(pos_msg)

    rospy.spin()


if __name__ == '__main__':
    head_init()

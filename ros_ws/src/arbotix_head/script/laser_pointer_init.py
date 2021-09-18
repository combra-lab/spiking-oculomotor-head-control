#!/usr/bin/env python

import rospy
from arbotix_msgs.srv import SetSpeed
from std_msgs.msg import Float64


def laser_pointer_init(init_pos):
    """
    Move laser pointer to init position

    Args:
        init_pos (list): init positions

    """

    rospy.init_node("init_laser_point_pos")
    speed_srv_list = []
    pos_pub_list = []
    name_list = ['laser_pan', 'laser_tilt']
    for name in name_list:
        speed_srv_name = '/laser/' + name + '/set_speed'
        speed_srv_list.append(rospy.ServiceProxy(speed_srv_name, SetSpeed))
        pos_pub_name = '/laser/' + name + '/command'
        pos_pub_list.append(rospy.Publisher(pos_pub_name, Float64, queue_size=5))

    for speed_srv in speed_srv_list:
        try:
            speed_srv(2.0)
        except rospy.ServiceException as e:
            print("Set Speed Failed ...")

    for ii, pos_pub in enumerate(pos_pub_list, 0):
        pos_msg = Float64()
        pos_msg.data = init_pos[ii]
        pos_pub.publish(pos_msg)

    rospy.spin()


if __name__ == '__main__':
    laser_pointer_init([0.0, 0.0])

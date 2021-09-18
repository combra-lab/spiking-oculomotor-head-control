import rospy
import numpy as np
import math
from arbotix_msgs.srv import SetSpeed
from std_msgs.msg import Float64


def laser_point_controller(servo_position_list, servo_speed, max_ros_ita, ros_rate):
    """
    Laser point controller for experiments

    Args:
        servo_position_list (Numpy Array): list of servo positions for each ROS step
        servo_speed (float): servo speed
        max_ros_ita (int): max steps for ROS
        ros_rate (int): ROS update rate

    """
    rospy.init_node("laser_point_control")
    speed_srv_list = []
    pos_pub_list = []
    name_list = ['laser_pan', 'laser_tilt']
    for name in name_list:
        speed_srv_name = '/laser/' + name + '/set_speed'
        speed_srv_list.append(rospy.ServiceProxy(speed_srv_name, SetSpeed))
        pos_pub_name = '/laser/' + name + '/command'
        pos_pub_list.append(rospy.Publisher(pos_pub_name, Float64, queue_size=1))

    for speed_srv in speed_srv_list:
        try:
            speed_srv(servo_speed)
        except rospy.ServiceException as e:
            print("Set Speed Failed ...")

    rate = rospy.Rate(ros_rate)
    ros_ita = 0
    while (not rospy.is_shutdown()) and (ros_ita < max_ros_ita):
        for ii, pos_pub in enumerate(pos_pub_list, 0):
            pos_msg = Float64()
            pos_msg.data = servo_position_list[ii, ros_ita]
            pos_pub.publish(pos_msg)
        ros_ita += 1
        rate.sleep()


def test_multiple_circle(circle_num):
    """
    Test multiple circle in fixed time

    Args:
        circle_num (int): number of circles

    """
    ros_rate = 20
    ros_max_ita = 20 * 60

    single_circle_ita = ros_max_ita // circle_num
    servo_pos_list = generate_single_circle_path(single_circle_ita, 0.2)
    all_servo_pos_list = np.zeros((2, ros_max_ita))
    for num in range(circle_num):
        all_servo_pos_list[:, num*single_circle_ita:(num+1)*single_circle_ita] = servo_pos_list

    servo_spd = 0.4 * circle_num
    laser_point_controller(all_servo_pos_list, servo_spd, ros_max_ita, ros_rate)


def test_chirp(max_circle_time, decrease_rate):
    """
    Test chirp signal with different speed circles

    Args:
        max_circle_time (int): max time for one circle in seconds
        decrease_rate (float): decrease rate for circle time
    """
    ros_rate = 20
    ros_max_ita = 0
    servo_pos_list = np.zeros((2, 0))
    circle_time = max_circle_time

    while circle_time >= 1:
        ros_max_ita += int(circle_time * ros_rate)
        single_circle_pos_list = generate_single_circle_path(int(circle_time*ros_rate), 0.2)
        servo_pos_list = np.concatenate((servo_pos_list, single_circle_pos_list), axis=1)

        circle_time = circle_time / decrease_rate

    print("Overall time: ", ros_max_ita)
    servo_spd = 5.0
    laser_point_controller(servo_pos_list, servo_spd, ros_max_ita, ros_rate)


def test_step():
    """
    Test step function
    """
    ros_rate = 20
    ros_max_ita = 20 * 32

    laser_z = 0.086 + 0.043 + 0.035
    step_point_list = [[0.2, laser_z], [0.2, laser_z+0.2], [0.2, laser_z+0.4],
                       [0.0, laser_z+0.2], [-0.2, laser_z+0.4],
                       [-0.2, laser_z+0.2], [-0.2, laser_z], [0, laser_z]]
    step_point_time = 4 * ros_rate

    servo_pos_list = np.zeros((2, ros_max_ita))
    for ii, step_point in enumerate(step_point_list):
        tmp_pan, tmp_tilt = compute_laser_pointer_inverse_kinematic(step_point[0], step_point[1])
        servo_pos_list[0, ii*step_point_time:(ii+1)*step_point_time] = tmp_pan
        servo_pos_list[1, ii * step_point_time:(ii + 1) * step_point_time] = -tmp_tilt

    servo_spd = 4.0
    laser_point_controller(servo_pos_list, servo_spd, ros_max_ita, ros_rate)


def compute_laser_pointer_inverse_kinematic(x_pos, z_pos,
                                            base_link_2_wall_y=-(0.3 + 0.191 * 1.5),
                                            base_link_2_laser_base_y=-0.191,
                                            laser_base_2_laser_pan_z=0.086,
                                            laser_pan_2_laser_tilt_z=0.043,
                                            laser_tilt_2_laser_z=0.035):
    """
    Compute laser pointer pan and tilt degree for a certain position on the wall

    Args:
        x_pos (float): x axis position on the wall
        z_pos (float): y axis position on the wall
        base_link_2_wall_y (float): base_link to wall on y axis
        base_link_2_laser_base_y (float): base_link to laser_base on y axis
        laser_base_2_laser_pan_z (float): laser_base to laser_pan on z axis
        laser_pan_2_laser_tilt_z (float): laser_pan to laser_tilt on z axis
        laser_tilt_2_laser_z (float): laser_tilt to laser on z axis

    Returns:
        pan_pos: pan degree
        tilt_pos: tilt degree

    """
    laser_base_2_wall_y = base_link_2_wall_y - base_link_2_laser_base_y
    alpha = -(x_pos / laser_base_2_wall_y)
    cos_pan_pos = math.sqrt(1.0 / (alpha**2 + 1))
    if x_pos > 0:
        pan_pos = math.acos(cos_pan_pos)
    else:
        pan_pos = -math.acos(cos_pan_pos)

    beta = (z_pos - laser_base_2_laser_pan_z - laser_pan_2_laser_tilt_z) * cos_pan_pos / laser_base_2_wall_y
    gamma = -(laser_tilt_2_laser_z * cos_pan_pos) / laser_base_2_wall_y
    a = 1.0 + beta**2
    b = 2.0 * beta * gamma
    c = gamma**2 - 1
    cos_tilt_pos = (-b + math.sqrt(b**2 - 4.0*a*c)) / (2.0 * a)
    if z_pos > (laser_base_2_laser_pan_z + laser_pan_2_laser_tilt_z + laser_tilt_2_laser_z):
        tilt_pos = math.acos(cos_tilt_pos)
    else:
        tilt_pos = -math.acos(cos_tilt_pos)

    return pan_pos, tilt_pos


def compute_laser_pointer_forward_kinematic(pan_pos, tilt_pos,
                                            base_link_2_wall_y=-(0.3 + 0.191 * 1.5),
                                            base_link_2_laser_base_y=-0.191,
                                            laser_base_2_laser_pan_z=0.086,
                                            laser_pan_2_laser_tilt_z=0.043,
                                            laser_tilt_2_laser_z=0.035):
    """
    Compute laser pointer wall position base on pan and tilt degrees

    Args:
        pan_pos (float): pan degree
        tilt_pos (float): tilt degree
        base_link_2_wall_y (float): base_link to wall on y axis
        base_link_2_laser_base_y (float): base_link to laser_base on y axis
        laser_base_2_laser_pan_z (float): laser_base to laser_pan on z axis
        laser_pan_2_laser_tilt_z (float): laser_pan to laser_tilt on z axis
        laser_tilt_2_laser_z (float): laser_tilt to laser on z axis

    Returns:
        wall_laser_point: wall laser point

    """
    # Compute transformation matrix for each link
    base_link_2_laser_pan_trans = np.array([[math.cos(pan_pos), -math.sin(pan_pos), 0., 0.],
                                            [math.sin(pan_pos), math.cos(pan_pos), 0.,
                                             base_link_2_laser_base_y],
                                            [0., 0., 1., laser_base_2_laser_pan_z],
                                            [0., 0., 0., 1.]])

    laser_pan_2_laser_tilt_trans = np.array([[1., 0., 0., 0.],
                                             [0., math.cos(tilt_pos), -math.sin(tilt_pos), 0.],
                                             [0., math.sin(tilt_pos), math.cos(tilt_pos),
                                              laser_pan_2_laser_tilt_z],
                                             [0., 0., 0., 1.]])

    laser_tilt_2_laser_trans = np.array([[1., 0., 0., 0.],
                                         [0., 1., 0., 0.],
                                         [0., 0., 1., laser_tilt_2_laser_z],
                                         [0., 0., 0., 1.]])

    laser_2_wall_y = (base_link_2_wall_y - base_link_2_laser_base_y) / (
            math.cos(pan_pos) * math.cos(tilt_pos)) + (
                             math.sin(tilt_pos) * laser_tilt_2_laser_z) / math.cos(tilt_pos)
    laser_2_wall_trans = np.array([[1., 0., 0., 0.],
                                   [0., 1., 0., laser_2_wall_y],
                                   [0., 0., 1., 0.],
                                   [0., 0., 0., 1.]])

    # Compute final transformation from base link to laser point on the wall
    base_link_2_wall_laser_point_trans = np.matmul(np.matmul(np.matmul(base_link_2_laser_pan_trans,
                                                                       laser_pan_2_laser_tilt_trans),
                                                             laser_tilt_2_laser_trans),
                                                   laser_2_wall_trans)
    wall_laser_point = base_link_2_wall_laser_point_trans[:3, 3].tolist()

    return wall_laser_point


def generate_single_circle_path(circle_sample_points, circle_r):
    """
    Generate laser points pan and tilt path for a single circle

    Args:
        circle_sample_points (int): number of steps for complete the circle
        circle_r (float): radius of circle in meters

    Returns:
        servo_pos_list: list of servo positions

    """
    laser_z = 0.086 + 0.043 + 0.035
    theta = np.linspace(0, 2 * np.pi, circle_sample_points)
    x_pos_list = circle_r * np.cos(theta-np.pi/2.0)
    z_pos_list = circle_r * np.sin(theta-np.pi/2.0) + laser_z + circle_r

    servo_pos_list = np.zeros((2, circle_sample_points))
    for num in range(circle_sample_points):
        tmp_pan, tmp_tilt = compute_laser_pointer_inverse_kinematic(x_pos_list[num], z_pos_list[num])
        servo_pos_list[0, num] = tmp_pan
        servo_pos_list[1, num] = -tmp_tilt

    return servo_pos_list


if __name__ == '__main__':
    from params import cfg
    if cfg['laser']['exp'] == 'step':
        test_step()
    elif cfg['laser']['exp'] == 'circle':
        test_multiple_circle(cfg['laser']['circle_num'])
    elif cfg['laser']['exp'] == 'chirp':
        test_chirp(cfg['laser']['chirp_max_circle_time'], cfg['laser']['chirp_decrease_rate'])
    else:
        print("Wrong Experiment Param: Please Enter Step, Circle, or Chirp ...")


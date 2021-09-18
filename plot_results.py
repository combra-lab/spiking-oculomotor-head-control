import numpy as np
import matplotlib.pyplot as plt
import math


def plot_and_analyze_exp_results(laser_point_list, left_point_list, right_point_list,
                                 ros_rate=20, distance_scale=10.0,
                                 head_2_way_y=0.586, eye_2_way_z=0.295, use_degree=False):
    """
    Plot and analyze the experiment results

    Args:
        laser_point_list (Numpy array): list of recorded laser point positions
        left_point_list (Numpy array): list of recorded left point positions
        right_point_list (Numpy array): list of recorded right point positions
        ros_rate (int): ros update rate
        distance_scale (float): distance scale of the recorded positions
        head_2_way_y (float): distance from head to wall
        eye_2_way_z (float): height of eye
        use_degree (bool): if true, use eye moving degree for time plots and errors

    Returns:
        laser_distance: distance of laser moving
        eye_distance_list: distances of eye moving
        eye_error_list: error of eye moving against laser

    """
    ros_ita = laser_point_list.shape[0]
    laser_point_list = laser_point_list / distance_scale
    left_point_list = left_point_list / distance_scale
    right_point_list = right_point_list / distance_scale

    fig, ax = plt.subplots(3, 1)
    ax[0].set_title("Project trajectory on the wall")
    ax[0].set_xlabel("X axis (meter)")
    ax[0].set_ylabel("Y axis (meter)")
    ax[0].plot(laser_point_list[:, 0], laser_point_list[:, 2], 'k')
    ax[0].plot(left_point_list[:, 0], left_point_list[:, 2], 'r')
    ax[0].plot(right_point_list[:, 0], right_point_list[:, 2], 'b')
    ax[0].legend(["laser", "left", "right"])

    def compute_point_distance(point_list):
        """
        Compute overall moving distance of a point list

        Args:
            point_list (Numpy array): list of positions

        Returns:
            distance: overall distance

        """
        delta_x_list = point_list[1:, 0] - point_list[:ros_ita - 1, 0]
        delta_y_list = point_list[1:, 2] - point_list[:ros_ita - 1, 2]
        delta_list = np.sqrt(delta_x_list ** 2 + delta_y_list ** 2)
        distance = np.sum(delta_list)
        return distance

    def compute_point_error(eye_point_list, target_point_list):
        """
        Compute absolute error between eye and laser

        Args:
            eye_point_list (Numpy array): list of eye positions
            target_point_list (Numpy array): list of laser positions

        Returns:
            error_mean: mean position error
            error_std: STD position error

        """
        error_x_list = eye_point_list[:, 0] - target_point_list[:, 0]
        error_y_list = eye_point_list[:, 2] - target_point_list[:, 2]
        error_list = np.sqrt(error_x_list ** 2 + error_y_list ** 2)
        error_mean = np.mean(error_list)
        error_std = np.std(error_list)
        return error_mean, error_std

    laser_distance = compute_point_distance(laser_point_list)
    eye_distance_list = [compute_point_distance(left_point_list), compute_point_distance(right_point_list)]

    plot_x_axis = np.arange(ros_ita) * (1000 / ros_rate)

    ax[1].set_title("Project X axis trajectory against time")
    ax[1].set_ylabel("X axis (meter)")
    ax[1].set_xlabel("Time (ms)")
    ax[1].plot(plot_x_axis, laser_point_list[:, 0], 'k')
    ax[1].plot(plot_x_axis, left_point_list[:, 0], 'r')
    ax[1].plot(plot_x_axis, right_point_list[:, 0], 'b')
    ax[1].legend(["laser", "left", "right"])

    ax[2].set_title("Project Y axis trajectory against time")
    ax[2].set_ylabel("Y axis (meter)")
    ax[2].set_xlabel("Time (ms)")
    ax[2].plot(plot_x_axis, laser_point_list[:, 2], 'k')
    ax[2].plot(plot_x_axis, left_point_list[:, 2], 'r')
    ax[2].plot(plot_x_axis, right_point_list[:, 2], 'b')
    ax[2].legend(["laser", "left", "right"])

    eye_error_list = [compute_point_error(left_point_list, laser_point_list),
                      compute_point_error(right_point_list, laser_point_list)]

    return laser_distance, eye_distance_list, eye_error_list


if __name__ == '__main__':
    import pickle

    with open('./exp_results/loihi_step_exp.p', 'rb') as fw:
        saved_data = pickle.load(fw)
        laser_list = saved_data[0][:, :]
        left_list = saved_data[1][:, :]
        right_list = saved_data[2][:, :]

    laser_dis, eye_dis_list, eye_err_list = plot_and_analyze_exp_results(laser_list,
                                                                          left_list,
                                                                          right_list)

    print("Laser Travel Distance: ", laser_dis, " Eye Travel Distance (Left, Right): ", eye_dis_list)
    print("Left Eye Error (Mean, STD): ", eye_err_list[0])
    print("Right Eye Error (Mean, STD): ", eye_err_list[1])

    plt.show()

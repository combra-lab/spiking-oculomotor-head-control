import numpy as np
import nxsdk.api.n2a as nx
from nxsdk.graph.processes.phase_enums import Phase
import os
from snn_loihi.oculumotor_snn import RobotHeadNet


def setup_full_head_snn():
    """
    Setup Full Head SNN on Loihi for testing

    Returns:
        net: Loihi network object
        input_conn_dict: dictionary of input fake connections

    """
    joint_name_list = ['eye_pan', 'eye_left_tilt', 'eye_right_tilt']

    ebn_2_eye_motor_conn_mask_dict = {'eye_pan': np.array([[1, 0], [0, 1], [0, 0],
                                                           [0, 0], [0, 0], [0, 0]]),
                                      'eye_left_tilt': np.array([[0, 0], [0, 0], [1, 0],
                                                                 [0, 1], [0, 0], [0, 0]]),
                                      'eye_right_tilt': np.array([[0, 0], [0, 0], [0, 0],
                                                                  [0, 0], [1, 0], [0, 1]])}

    llbn_2_neck_motor_conn_mask_dict = {'eye_pan': np.array([[1, 0], [0, 1], [0, 0], [0, 0]]),
                                        'eye_left_tilt': np.array([[0, 0], [0, 0], [1, 0], [0, 0]]),
                                        'eye_right_tilt': np.array([[0, 0], [0, 0], [0, 0], [0, 1]])}

    ebn_2_coupling_conn_mask_dict = {'eye_left_tilt': np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]),
                                     'eye_right_tilt': np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])}

    net = nx.NxNet()
    eye_motor_neuron, neck_motor_neuron = RobotHeadNet.motor_neurons(net)
    input_neuron_dict, input_conn_dict = RobotHeadNet.online_input_neurons(net, joint_name_list)
    ebn_dict, llbn_dict = RobotHeadNet.control_core_module_all_joints(net, input_neuron_dict, joint_name_list)
    RobotHeadNet.eye_joints_control(eye_motor_neuron, ebn_dict, ebn_2_eye_motor_conn_mask_dict)
    RobotHeadNet.neck_joints_control(neck_motor_neuron, ebn_dict, llbn_2_neck_motor_conn_mask_dict)
    RobotHeadNet.eye_coupling_control(net, eye_motor_neuron, ebn_dict, ebn_2_coupling_conn_mask_dict)
    RobotHeadNet.online_motor_neurons_spike_probe(eye_motor_neuron, neck_motor_neuron)

    return net, input_conn_dict


def compile_single_joint_head_snn(net, input_conn_dict, snip_path="./snn_loihi/snips"):
    """
    Compile Loihi network with online encoding and decoding

    Args:
        net (NxNet): Loihi network object
        input_conn_dict (dict): dictionary of input fake connections
        snip_path (str): directory for snip

    Returns:
        board: Loihi compiled network
        encoder_channel: encoder channel
        decoder_channel: decoder channel

    """
    compiler = nx.N2Compiler()
    board = compiler.compile(net)
    input_neuron_id = RobotHeadNet.online_get_fake_input_connection_axon_id(net, input_conn_dict)
    print("Input Neuron Axon Id: ", input_neuron_id)
    include_dir = os.path.abspath(snip_path)
    encoder_snip = board.createSnip(
        Phase.EMBEDDED_SPIKING,
        name="encoder",
        includeDir=include_dir,
        cFilePath=include_dir + "/encoder.c",
        funcName="run_encoder",
        guardName="do_encoder"
    )
    decoder_snip = board.createSnip(
        Phase.EMBEDDED_MGMT,
        name="decoder",
        includeDir=include_dir,
        cFilePath=include_dir + "/decoder.c",
        funcName="run_decoder",
        guardName="do_decoder"
    )
    encoder_channel = board.createChannel(b'encodeinput', "int", 6)
    encoder_channel.connect(None, encoder_snip)
    decoder_channel = board.createChannel(b'decodeoutput', "int", 10)
    decoder_channel.connect(decoder_snip, None)
    return board, encoder_channel, decoder_channel


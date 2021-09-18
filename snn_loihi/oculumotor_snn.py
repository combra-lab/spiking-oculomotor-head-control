import nxsdk.api.n2a as nx
from nxsdk.graph.monitor.probes import SpikeProbeCondition
import numpy as np


class RobotHeadNet:
    """ SNN for robot head control on Loihi """

    @staticmethod
    def motor_neurons(net):
        """
        Create motor neurons for eyes and necks

        Args:
            net (NxNet): Loihi network object

        Returns:
            eye_motor_neuron: motor neurons for eyes
            neck_motor_neuron: motor neurons for neck

        """
        motor_neuron_prototype = nx.CompartmentPrototype(compartmentVoltageDecay=1,
                                                         compartmentCurrentDecay=4095,
                                                         vThMant=20,
                                                         functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE)
        # Create 6 motor neurons for eyes
        # 0: Up, 1: Down, 2: Left eye left, 3: Left eye right, 4: Right eye left, 5: Right eye right
        eye_motor_neuron = net.createCompartmentGroup(size=6, prototype=motor_neuron_prototype)

        # Create 4 motor neurons for neck
        # 0: Up, 1: Down, 2: Left, 3: Right
        neck_motor_neuron = net.createCompartmentGroup(size=4, prototype=motor_neuron_prototype)

        return eye_motor_neuron, neck_motor_neuron

    @classmethod
    def control_core_module_all_joints(cls, net, input_neuron_dict, joint_name_list):
        """
        Create control core module for all control joints

        Args:
            net (NxNet): Loihi network object
            input_neuron_dict (dict): dictionary for input neuron groups
            joint_name_list (list): list of joint names

        Returns:
            ebn_neuron_dict (dict): dictionary for excitatory bursting neuron groups
            llbn_soma_dict (dict): dictionary for LLBN soma

        """
        # Define general prototype for ebn neuron, llbn input neuron, and ifn neuron
        nrn_prototype = nx.CompartmentPrototype(compartmentVoltageDecay=1,
                                                compartmentCurrentDecay=4095,
                                                vThMant=10,
                                                functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE)

        # Define neuron prototypes (bn_prototype, inh_prototype) for llbn bursting neuron
        rp = nx.CompartmentPrototype(compartmentVoltageDecay=20 * int(1 / 1500 * 2 ** 12),
                                     compartmentCurrentDecay=int(1 / 2 * 2 ** 12),
                                     thresholdBehavior=2,
                                     vThMant=1000)
        lp = nx.CompartmentPrototype(compartmentVoltageDecay=20 * int(1 / 1500 * 2 ** 12),
                                     compartmentCurrentDecay=int(1 / 2 * 2 ** 12),
                                     compartmentJoinOperation=0,
                                     thresholdBehavior=2,
                                     vThMant=100 * 120)
        root_proto = nx.CompartmentPrototype(compartmentCurrentDecay=4095,
                                             compartmentVoltageDecay=300 * int(1 / 1500 * 2 ** 12),
                                             thresholdBehavior=2,
                                             vThMant=120)
        soma_proto = nx.CompartmentPrototype(compartmentCurrentDecay=4095,
                                             compartmentVoltageDecay=0)
        soma_proto.addDendrite(root_proto, nx.COMPARTMENT_JOIN_OPERATION.OR)
        root_proto.addDendrite([lp, rp], nx.COMPARTMENT_JOIN_OPERATION.PASS)
        bn_prototype = nx.NeuronPrototype(soma_proto)
        inh_soma_proto = nx.CompartmentPrototype(vThMant=1500,
                                                 compartmentVoltageDecay=int(1 / 500 * 2 ** 12),
                                                 functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE)
        inh_prototype = nx.NeuronPrototype(inh_soma_proto)

        # Construct control core module for each joint
        ebn_neuron_dict = dict()
        llbn_soma_dict = dict()

        for joint_name in joint_name_list:
            ebn_neuron, llbn_soma = cls.control_core_module_single_joint(net, input_neuron_dict[joint_name],
                                                                         nrn_prototype, bn_prototype, inh_prototype)
            ebn_neuron_dict[joint_name] = ebn_neuron
            llbn_soma_dict[joint_name] = llbn_soma

        return ebn_neuron_dict, llbn_soma_dict

    @staticmethod
    def control_core_module_single_joint(net, input_neuron, nrn_prototype, bn_prototype, inh_prototype):
        """
        Create control core module for single control joint

        Args:
            net (NxNet): Loihi network object
            input_neuron (CompartmentGroup): input neuron group
            nrn_prototype (CompartmentPrototype): compartment prototype for general neurons
            bn_prototype (NeuronPrototype): neuron prototype for bursting neuron
            inh_prototype (NeuronPrototype): neuron prototype for bursting neuron

        Returns:
            ebn_neuron: excitatory bursting neuron group
            llbn_soma: soma of LLBN neuron

        """
        # Create excitatory bursting neuron and connection input neuron
        ebn_neuron = net.createCompartmentGroup(size=2, prototype=nrn_prototype)
        input_neuron.connect(ebn_neuron,
                             prototype=nx.ConnectionPrototype(numWeightBits=8),
                             weight=np.eye(2) * 12,
                             connectionMask=np.int_(np.eye(2)))

        # Create LLBN input neuron and connection input neuron
        llbn_input_neuron = net.createCompartmentGroup(size=2, prototype=nrn_prototype)
        input_neuron.connect(llbn_input_neuron,
                             prototype=nx.ConnectionPrototype(numWeightBits=8),
                             weight=np.eye(2) * 6,
                             connectionMask=np.int_(np.eye(2)))

        # Create IFN neuron
        ifn_neuron = net.createCompartmentGroup(size=2, prototype=nrn_prototype)

        # Create LLBN neuron
        llbn_neuron = net.createNeuronGroup(size=2, prototype=bn_prototype)
        llbn_soma = llbn_neuron.soma
        llbn_root = llbn_neuron.dendrites[0]
        llbn_l = llbn_neuron.dendrites[0].dendrites[0]
        llbn_r = llbn_neuron.dendrites[0].dendrites[1]

        llbn_inh_neuron = net.createNeuronGroup(size=2, prototype=inh_prototype)
        llbn_inh_soma = llbn_inh_neuron.soma

        llbn_soma.connect(llbn_inh_soma,
                          prototype=nx.ConnectionPrototype(weightExponent=-1),
                          weight=np.eye(2) * 4,
                          connectionMask=np.int_(np.eye(2)))
        llbn_inh_soma.connect(llbn_l,
                              prototype=nx.ConnectionPrototype(weightExponent=2),
                              weight=np.eye(2) * (-256),
                              connectionMask=np.int_(np.eye(2)))

        # Connect LLBN input neuron with LLBN neuron
        llbn_input_neuron.connect(llbn_r,
                                  prototype=nx.ConnectionPrototype(),
                                  weight=np.eye(2) * 40,
                                  connectionMask=np.int_(np.eye(2)))
        llbn_input_neuron.connect(llbn_l,
                                  prototype=nx.ConnectionPrototype(),
                                  weight=np.eye(2) * 40,
                                  connectionMask=np.int_(np.eye(2)))

        # Connect LLBN neuron to EBN neuron
        llbn_soma.connect(ebn_neuron,
                          prototype=nx.ConnectionPrototype(),
                          weight=np.eye(2) * 20,
                          connectionMask=np.int_(np.eye(2)))

        # Connect EBN neuron with IFN neuron and inhibit LLBN neuron
        ebn_neuron.connect(ifn_neuron,
                           prototype=nx.ConnectionPrototype(),
                           weight=np.eye(2) * 40,
                           connectionMask=np.int_(np.eye(2)))
        ifn_neuron.connect(llbn_l,
                           prototype=nx.ConnectionPrototype(),
                           weight=np.eye(2) * (-12),
                           connectionMask=np.int_(np.eye(2)))
        ifn_neuron.connect(llbn_r,
                           prototype=nx.ConnectionPrototype(),
                           weight=np.eye(2) * (-12),
                           connectionMask=np.int_(np.eye(2)))

        return ebn_neuron, llbn_soma

    @staticmethod
    def eye_joints_control(eye_motor_neuron, ebn_neuron_dict, ebn_2_eye_motor_conn_mask_dict):
        """
        Create connections for eye joints control

        Args:
            eye_motor_neuron (CompartmentGroup): motor neurons for eyes
            ebn_neuron_dict (dict): dictionary for EBN neuron groups
            ebn_2_eye_motor_conn_mask_dict (dict): dictionary for EBN to eye motor neuron connection mask

        """
        for key in ebn_neuron_dict:
            conn_mask = ebn_2_eye_motor_conn_mask_dict[key]
            ebn_neuron_dict[key].connect(eye_motor_neuron,
                                         prototype=nx.ConnectionPrototype(),
                                         weight=conn_mask * 30,
                                         connectionMask=np.int_(conn_mask))

    @staticmethod
    def neck_joints_control(neck_motor_neuron, llbn_soma_dict, llbn_2_neck_motor_conn_mask_dict):
        """
        Create connections for neck joints control

        Args:
            neck_motor_neuron (CompartmentGroup): motor neurons for neck
            llbn_soma_dict (dict): dictionary for LLBN soma groups
            llbn_2_neck_motor_conn_mask_dict (dict): dictionary for LLBN to neck motor neuron connection mask

        """
        for key in llbn_soma_dict:
            conn_mask = llbn_2_neck_motor_conn_mask_dict[key]
            llbn_soma_dict[key].connect(neck_motor_neuron,
                                        prototype=nx.ConnectionPrototype(numWeightBits=8),
                                        weight=conn_mask * 8,
                                        connectionMask=np.int_(conn_mask))

    @staticmethod
    def eye_coupling_control(net, eye_motor_neuron, ebn_neuron_dict, ebn_2_coupling_conn_mask_dict):
        """
        Create sub-network for coupling movement of two eyes

        Args:
            net (NxNet): Loihi network object
            eye_motor_neuron (CompartmentGroup): motor neurons for eyes
            ebn_neuron_dict (dict): dictionary for EBN neuron groups
            ebn_2_coupling_conn_mask_dict (dict): dictionary for EBN to coupling neuron connection mask

        """
        # Create coupling neurons
        # Coupling neuron, 0: Left eye left > Right eye left, 1: Left eye left < Right eye left
        # 2: Left eye right > Right eye right, 3: Left eye right < Right eye right
        nrn_prototype = nx.CompartmentPrototype(compartmentVoltageDecay=500,
                                                compartmentCurrentDecay=4095,
                                                vThMant=10,
                                                functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE)
        coupling_neuron = net.createCompartmentGroup(size=4, prototype=nrn_prototype)

        # Connection EBN neurons to coupling neurons
        for key in ebn_2_coupling_conn_mask_dict:
            conn_mask = ebn_2_coupling_conn_mask_dict[key]
            ebn_neuron_dict[key].connect(coupling_neuron,
                                         prototype=nx.ConnectionPrototype(),
                                         weight=conn_mask * 20,
                                         connectionMask=np.int_(np.abs(conn_mask)))

        # Connection coupling neurons to eye motor neurons
        # Coupling neuron to motor neuron:
        # 0: Right eye left, 1: Left, Right eye left, 2: Left, Right eye right, 3: Left eye right
        conn_w = np.array([[0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 12, 0, 0],
                           [0, 0, 0, 12],
                           [12, 0, 0, 0],
                           [0, 0, 12, 0]])
        conn_mask = np.int_(np.array([[0, 0, 0, 0],
                                      [0, 0, 0, 0],
                                      [0, 1, 0, 0],
                                      [0, 0, 0, 1],
                                      [1, 0, 0, 0],
                                      [0, 0, 1, 0]]))
        coupling_neuron.connect(eye_motor_neuron,
                                prototype=nx.ConnectionPrototype(numWeightBits=8),
                                weight=conn_w,
                                connectionMask=conn_mask)

    @staticmethod
    def online_input_neurons(net, joint_name_list):
        """
        Online input neurons and create fake connections for online input encoding

        Args:
            net (NxNet): Loihi network object
            joint_name_list (list): list of joint names

        Returns:
            input_neuron_dict: dictionary for input neuron groups
            input_neuron_conn_dict: dictionary for fake connection to input neuron groups

        """
        input_neuron_dict = dict()
        input_neuron_conn_dict = dict()

        for joint_name in joint_name_list:
            input_neuron = net.createCompartmentGroup(size=2, prototype=nx.CompartmentPrototype())
            pseudo_input_neuron = net.createCompartmentGroup(size=2, prototype=nx.CompartmentPrototype())
            input_neuron_conn = pseudo_input_neuron.connect(input_neuron,
                                                            prototype=nx.ConnectionPrototype(),
                                                            weight=np.eye(2) * 120,
                                                            connectionMask=np.int_(np.eye(2)))
            input_neuron_dict[joint_name] = input_neuron
            input_neuron_conn_dict[joint_name] = input_neuron_conn

        return input_neuron_dict, input_neuron_conn_dict

    @staticmethod
    def online_get_fake_input_connection_axon_id(net, input_neuron_conn_dict):
        """
        Get axon id for fake connections to input neurons for online input encoding

        Args:
            net (NxNet): Loihi network object
            input_neuron_conn_dict (dict): dictionary for fake connection to input neuron groups

        Returns:
            input_neuron_id_dict: dictionary for axon id of input neurons

        """
        input_neuron_id_dict = dict()
        for key in input_neuron_conn_dict:
            for ii, conn in enumerate(input_neuron_conn_dict[key], 0):
                input_neuron_id_dict[key + str(ii)] = net.resourceMap.inputAxon(conn.inputAxon.nodeId)

        return input_neuron_id_dict

    @staticmethod
    def online_motor_neurons_spike_probe(eye_motor_neuron, neck_motor_neuron):
        """
        Online controller spike probe
        Args:
            eye_motor_neuron (CompartmentGroup): motor neuron for eyes
            neck_motor_neuron (CompartmentGroup): motor neuron for neck

        """
        custom_probe_cond = SpikeProbeCondition(tStart=10000000000)
        eye_motor_neuron.probe(nx.ProbeParameter.SPIKE, custom_probe_cond)
        neck_motor_neuron.probe(nx.ProbeParameter.SPIKE, custom_probe_cond)


from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype
import numpy as np

class LSTMPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None # None: no recurrent network, Int num: MLP

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

        share_layer = obz

        for i in range(num_hid_layers):
            share_layer = tf.nn.tanh(tf.layers.dense(share_layer, hid_size, name="fc%i" % (i + 1),
                                                       kernel_initializer=U.normc_initializer(1.0)))

        self.lstm_input = tf.expand_dims(share_layer, [0])

        step_size = tf.shape(obz)[:1]

        # add the recurrent layers or combined with one Mlp
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(hid_size, state_is_tuple=True)
        c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
        h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
        self.lstm_state_init = [c_init, h_init]

        c_in = U.get_placeholder(name="cin", dtype=tf.float32, shape=[1, lstm_cell.state_size.c])
        h_in = U.get_placeholder(name="hin", dtype=tf.float32, shape=[1, lstm_cell.state_size.h])

        self.state_in = [c_in, h_in]
        lstm_state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)

        self.lstm_out1, lstm_state_out = tf.nn.dynamic_rnn(lstm_cell, self.lstm_input,
                                                           initial_state=lstm_state_in,
                                                           sequence_length=step_size,
                                                           time_major=False)

        lstm_c, lstm_h = lstm_state_out
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
        self.lstm_out = tf.reshape(self.lstm_out1, [-1, hid_size])

        with tf.variable_scope('vf'):

            self.last_out = self.lstm_out
            # for i in range(num_hid_layers):
            #     self.last_out = tf.nn.tanh(tf.layers.dense(self.last_out, hid_size, name="fc%i"%(i+1), kernel_initializer=U.normc_initializer(1.0)))
            self.vpred = tf.layers.dense(self.last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:,0]

        with tf.variable_scope('pol'):
            self.last_out_pol = self.lstm_out
            # for i in range(num_hid_layers):
            #     self.last_out_pol = tf.nn.tanh(tf.layers.dense(self.last_out_pol, hid_size, name='fc%i'%(i+1), kernel_initializer=U.normc_initializer(1.0)))
            if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                mean = tf.layers.dense(self.last_out_pol, pdtype.param_shape()[0]//2, name='final', kernel_initializer=U.normc_initializer(0.01))
                logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
                self.pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            else:
                self.pdparam = tf.layers.dense(self.last_out_pol, pdtype.param_shape()[0], name='final', kernel_initializer=U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(self.pdparam)

        # self.state_in = []
        # self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob, self.state_in[0], self.state_in[1]], [ac, self.vpred, self.state_out])

    def act(self, stochastic, ob, state_in):
        ac1, vpred1, states=  self._act(stochastic, ob[None], state_in[0], state_in[1])
        return ac1[0], vpred1[0], states
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []


class MlpPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        with tf.variable_scope('vf'):
            obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            last_out = obz
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name="fc%i"%(i+1), kernel_initializer=U.normc_initializer(1.0)))
            self.vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:,0]

        with tf.variable_scope('pol'):
            last_out = obz
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name='fc%i'%(i+1), kernel_initializer=U.normc_initializer(1.0)))
            if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                mean = tf.layers.dense(last_out, pdtype.param_shape()[0]//2, name='final', kernel_initializer=U.normc_initializer(0.01))
                logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            else:
                pdparam = tf.layers.dense(last_out, pdtype.param_shape()[0], name='final', kernel_initializer=U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred1 =  self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []


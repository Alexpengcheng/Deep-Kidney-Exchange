#Asynchronous Advantage Actor-Critic Method on Kidney Exchange

import numpy as np
import tensorflow as tf
import threading
import multiprocessing
import os
import gym
import gym_kidney
from time import sleep
import scipy.signal

# Environment parameters
homogeneous={
    'tau':7,
    'rate':25,
    'k':50,
    'p':0.05,
    'p_a':0.01
}
heterogeneous={
    'tau':7,
    'rate':25,
    'k':50,
    'p':0.05,
    'p_l':0.1,
    'p_h':0.1,
    'p_a':0.1
}
kidney={
    'rate':25,
    'k':50
}
MODEL=homogeneous
MODEL_NAME='homogeneous'

# Network parameters
n_input=MODEL['tau']**2+MODEL['tau'] #The length of embedding should be specified first since initialization of neural net needs this
n_hidden1=256
n_hidden2=256
n_hidden3=200
n_hidden4=200
n_output_action=2 #2 actions
n_output_value=1
Discount_factor=0.99#discount factor
load_model=False
Maxtrajectory=200
Maxtrain=1000
Clip_norm=40.0
model_path='./model/'+MODEL_NAME

# Functions used to transfer the data between the local and global network
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

# Define the Actor-Critic Network
class A3Cnetwork():
    def __init__(self, scope, trainer):
        with tf.variable_scope(scope):
            # Define the input layers, weights and biases of neural net
            self.x = tf.placeholder("float", [None, n_input])

            weights = {
                'h1': tf.Variable(tf.random_normal([n_input, n_hidden1], stddev=1./np.sqrt(n_input)), name='h1'),
                'h2': tf.Variable(tf.random_normal([n_hidden1, n_hidden2], stddev=1./np.sqrt(n_hidden1)), name='h2'),
                'h3': tf.Variable(tf.random_normal([n_hidden2, n_hidden3], stddev=1./np.sqrt(n_hidden2)), name='h3'),
                'h4': tf.Variable(tf.random_normal([n_hidden3, n_hidden4], stddev=1./np.sqrt(n_hidden3)), name='h3'),
                'out_action': tf.Variable(tf.random_normal([n_hidden4, n_output_action], stddev=1./np.sqrt(n_hidden4)),name='out_action'),
                'out_value': tf.Variable(tf.random_normal([n_hidden4, n_output_value], stddev=1./np.sqrt(n_hidden4)), name='out_value')
            }

            biases = {
                'b1': tf.Variable(tf.zeros([n_hidden1]), name='b1'),
                'b2': tf.Variable(tf.zeros([n_hidden2]), name='b2'),
                'b3': tf.Variable(tf.zeros([n_hidden3]), name='b3'),
                'b4': tf.Variable(tf.zeros([n_hidden4]), name='b4'),
                'b_out_action': tf.Variable(tf.zeros([n_output_action]), name='b_out_action'),
                'b_out_value': tf.Variable(tf.zeros([n_output_value]), 'b_out_value')
            }

            self.layer_1 = tf.add(tf.matmul(self.x, weights['h1']), biases['b1'])
            self.layer_1 = tf.nn.relu(self.layer_1)
            self.layer_2 = tf.add(tf.matmul(self.layer_1, weights['h2']), biases['b2'])
            self.layer_2 = tf.nn.relu(self.layer_2)
            self.layer_3 = tf.add(tf.matmul(self.layer_2, weights['h3']), biases['b3'])
            self.layer_3 = tf.nn.relu(self.layer_3)
            self.layer_4 = tf.add(tf.matmul(self.layer_3, weights['h4']), biases['b4'])
            self.layer_4 = tf.nn.relu(self.layer_4)

            # Action output layer with softmax
            # Two actions '0' no match '1' maxmatch
            self.Action_outlayer = tf.add(tf.matmul(self.layer_4, weights['out_action']), biases['b_out_action'])
            self.Action_outlayer = tf.nn.softmax(logits=self.Action_outlayer)

            # Value output layer with linear activation
            self.Value_outlayer = tf.add(tf.matmul(self.layer_4, weights['out_value']), biases['b_out_value'])

            # Create the loss function and gradient updating for networks of local workers
            if scope !='global':
                self.action_batch = tf.placeholder(shape=[None],dtype=tf.int32)
                # Onehot in tf: 0-[1,0] 1-[0,1]
                self.actions_onehot = tf.one_hot(self.action_batch, n_output_action, dtype=tf.float32)
                self.target_value_batch = tf.placeholder(shape=[None],dtype=tf.float32)
                self.advantages_batch=tf.placeholder(shape=[None],dtype=tf.float32)
                # Find the probabilities of real actions taken
                self.responsible_outputs = tf.reduce_sum(self.Action_outlayer * self.actions_onehot, [1])

                # Loss function
                self.value_loss=tf.reduce_sum(tf.square(self.target_value_batch-tf.reshape(self.Value_outlayer,[-1])))
                self.policy_loss=-tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages_batch)
                self.entropy=- tf.reduce_sum(self.Action_outlayer * tf.log(self.Action_outlayer))
                self.loss=0.25 * self.value_loss + self.policy_loss - self.entropy * 0.01

                # Calculate the gradient wrt the local tf.Variables
                local_variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_variables)
                self.var_norms = tf.global_norm(local_variables)

                # Clip the gradient with norm
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, Clip_norm)

                # Use the local gradient to update the global tf.Variables
                global_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(self.gradients, global_variables))


class Worker():
    # Define input environment=game
    def __init__(self, name, trainer, model_path, global_episodes):
        self.name= 'worker_'+str(name)
        self.number=name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)

        # Define the containers used to store trajectories
        self.action_buffer=[]
        self.state_buffer=[]
        self.reward_buffer=[]
        self.value_buffer=[]
        self.episode_rewards = []

        self.summary_writer = tf.summary.FileWriter("train_ff_"+MODEL_NAME + str(self.number))

        # Create the local A3Cnetwork
        self.local_AC=A3Cnetwork(self.name,trainer)
        self.update_local_ops = update_target_graph('global', self.name)

        # Create the RL environment
        self.env=gym.make('kidney-v0')
        self.env=gym_kidney.ConfigWrapper(self.env, MODEL_NAME, MODEL)

    # Calculate the Advantage, Discounted rewards and update the neural net
    def train(self, action, state, reward, value, bootstrap, sess):
        # Convert the
        action=np.array(action)
        state=np.array(state)
        # state=np.reshape(state,[Maxtrajectory,4])
        state = np.reshape(state, [-1, n_input])
        reward=np.array(reward)
        value=np.array(value)
        # value=np.reshape(value,[Maxtrajectory,1])
        value = np.reshape(value, [-1, 1])
        value=value[:,0]




        # Generated the Advantage and Discounted rewards
        self.reward2=np.asarray(np.ndarray.tolist(reward) + [bootstrap[0,0]])
        discounted_reward=discount(self.reward2,Discount_factor)[:-1]
        self.value2=np.asarray(np.ndarray.tolist(value) + [bootstrap[0,0]])
        advantages=reward+Discount_factor*self.value2[1:]-self.value2[:-1]
        advantages=discount(advantages,Discount_factor)


        # Update the global neural net with current data
        feed_dict={self.local_AC.x:state,
                   self.local_AC.action_batch:action,
                   self.local_AC.advantages_batch:advantages,
                   self.local_AC.target_value_batch:discounted_reward}

        ao,vo,vl,pl,en,los,vn,gn,_=sess.run([self.local_AC.Action_outlayer,
                  self.local_AC.Value_outlayer,
                  self.local_AC.value_loss,
                  self.local_AC.policy_loss,
                  self.local_AC.entropy,
                  self.local_AC.loss,
                  self.local_AC.var_norms,
                  self.local_AC.grad_norms,
                  self.local_AC.apply_grads],feed_dict=feed_dict)

        return vl/len(action), pl/len(action), en/len(action), los/len(action), vn/len(action), gn/len(action)

    def work(self, max_episode, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)

        print('Start the Worker'+str(self.number))

        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                total_rewards = 0
                total_trajections = 0
                # Interaction with the environment
                # The max training iteration is Maxtrain
                # The max trajectories collected before train
                for i in range(max_episode):
                    # Reset the Kidney graph
                    total_rewards = 0
                    total_trajections = 0
                    observation = self.env.reset()
                    observation = np.reshape(observation,[1,n_input])
                    # print observation
                    Discounts=Discount_factor
                    # Synchronize the local network and initialize the episode buffer
                    sess.run(self.update_local_ops)
                    action = []
                    state = []
                    reward = []
                    value = []

                    done = False

                    while done==False:
                        total_trajections=total_trajections+1
                        Discounts=Discounts*Discount_factor
                        out_action,out_value=sess.run([self.local_AC.Action_outlayer,self.local_AC.Value_outlayer],
                                                      feed_dict={self.local_AC.x:observation})

                        # Sample an action based on the action distribution
                        choices=np.array([0,1])
                        print out_action, out_value
                        a=np.random.choice(choices,p=out_action[0,:])

                        # Store the episodes as list in python
                        state.append(observation)
                        action.append(a)
                        value.append(out_value)

                        # Interact with the environment
                        observation, r, done, info = self.env.step(a)
                        observation = np.reshape(observation, [1, n_input])

                        reward.append(r)

                        total_rewards=total_rewards+r
                        # print done
                        # When meet the maxtrajectories update the neural net
                        # if len(action)==Maxtrajectory:
                        #     # Bootstrap the next Value from the observation
                        #     bootvalue=sess.run(self.local_AC.Value_outlayer,
                        #                        feed_dict={self.local_AC.x:observation})
                        #
                        #     # Use train method to update global network
                        #
                        #     vl,pl,en,los,vn,gn=self.train(action, state, reward, value, bootvalue, sess)
                        #
                        #     # Clear all the episode buffer
                        #     action = []
                        #     state = []
                        #     reward = []
                        #     value = []
                        #
                        #     # Update the local network
                        #     sess.run(self.update_local_ops)
                        #     print ('Model updated')

                        if done:
                            #*******only for cartploe*********
                            # Update the network with the rest episode buffer
                            if len(action)!=0:
                                boot=np.array([[0.0]])
                                vl, pl, en, los, vn, gn = self.train(action, state, reward, value, boot, sess)
                                print ('model updated')

                            #**********************************

                            print ('Episodes finished ')


                    # Periodically save the Model parameters and summary results
                    if episode_count%50==0 and self.name=='worker_0':
                        saver.save(sess,self.model_path+'/model-ff-'+MODEL_NAME+str(episode_count)+'.cptk')
                        print('Model Saved')
                    self.episode_rewards.append(total_rewards)
                    print ('rewards:',total_rewards)

                    if episode_count%5==0:
                        mean_rewards=np.mean(self.episode_rewards[-5:])
                        summary=tf.Summary()
                        summary.value.add(tag='Perf/Reward', simple_value=float(mean_rewards))
                        summary.value.add(tag='Losses/Value Loss', simple_value=float(vl))
                        summary.value.add(tag='Losses/Policy Loss', simple_value=float(pl))
                        summary.value.add(tag='Losses/Entropy', simple_value=float(en))
                        summary.value.add(tag='Losses/Grad Norm', simple_value=float(gn))
                        summary.value.add(tag='Losses/Var Norm', simple_value=float(vn))
                        self.summary_writer.add_summary(summary, i)

                        self.summary_writer.flush()
                    if self.name=='worker_0)':
                        sess.run(self.increment)
                    episode_count=episode_count+1
                coord.request_stop()

tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    rate = tf.train.exponential_decay(0.005, global_episodes, 100, 0.5)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
    # Generate the global network as the container
    master_network=A3Cnetwork('global',None)
    num_workers = multiprocessing.cpu_count()  # Set workers ot number of available CPU threads

    # Generate workers
    workers=[]
    for i in range(1):
        workers.append(Worker(i,trainer,model_path,global_episodes))
    saver=tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if load_model == True:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    # Set asynchronous workers in each thread
    worker_threads=[]
    for worker in workers:
        worker_work = lambda: worker.work(Maxtrain, sess, coord, saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)
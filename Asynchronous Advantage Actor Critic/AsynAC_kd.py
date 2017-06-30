#Asynchronous Advantage Actor-Critic Method on Kidney Exchange

import numpy as np
import tensorflow as tf

"""Document:
Input Arguments:
	st_1.csv: file that save the current state
	control_parameters.txt: file that control the training and running process
						1st line:'0' load the neural net tf.Variables from the file
							 '1' first time run the code, and need to initialize tf.Variables

						2nd line: represent the current trajectory number T, when T>Maxtrajectory, train neural net
						3rd line: Maxtrajectory: the batch size of data used to train the neural net
	obj_value.txt: Batch of rewards provided by the Java part

Output Arguments:
	action.txt, state.txt: Batch of states and actions mantained by the python file itself, used to train the neural net
	kdnnet.ckpt: Save the tf.Variables of the neural net
"""

#Read data from the graph embedding from st.csv
#The csv last item is NaN, I remove it
initial_state = np.genfromtxt('st_1.csv', delimiter=',')
length=initial_state.size
initial_state=initial_state[0:length]
initial_state=initial_state.reshape((1,length))

#network parameters
n_input=length
n_hidden1=20
n_hidden2=20
n_output_action=2 #2 actions
n_output_value=1
Dicount_factor=0.6 #discount factor

Load = np.loadtxt('control.txt')
First_time = Load[0] #'0' indicate not 1st time run and don't need to initialize the Variables
T = Load[1]#trajectory number
Maxtrajectory = Load[2]

#tf graph input
x = tf.placeholder("float", [1, n_input])
action_batch=tf.placeholder('float',[Maxtrajectory,1])
state_batch=tf.placeholder('float',[Maxtrajectory,n_input])
reward_batch=tf.placeholder('float',[Maxtrajectory,1])

#Create the model
#x: input graph embedding
#weights and biases: dictionary of neural net weights and biases
def ff_nnet(x):
	#Hidden layer with ReLU activation
	layer_1=tf.add(tf.matmul(x, weights['h1']), biases['b1'])
	layer_1=tf.nn.relu(layer_1)
	layer_2=tf.add(tf.matmul(layer_1, weights['h2']), biases['b1'])
	layer_2=tf.nn.relu(layer_2)

	#Action output layer with softmax
	Action_outlayer=tf.add(tf.matmul(layer_2, weights['out_action']), biases['b_out_action'])
	Action_outlayer=tf.nn.softmax(logits=Action_outlayer)

	#Value output layer with linear activation
	Value_outlayer=tf.add(tf.matmul(layer_2,weights['out_value']),biases['b_out_value'])

	return Action_outlayer, Value_outlayer


#Layers weight and biases initialization
weights={
	'h1': tf.Variable(tf.random_normal([n_input, n_hidden1]),name='h1'),
	'h2': tf.Variable(tf.random_normal([n_hidden1, n_hidden2]),name='h2'),
	'out_action': tf.Variable(tf.random_normal([n_hidden2, n_output_action]),name='out_action'),
	'out_value': tf.Variable(tf.random_normal([n_hidden2, n_output_value]),name='out_value')
}

biases={
	'b1': tf.Variable(tf.random_normal([n_hidden1]),name='b1'),
	'b2': tf.Variable(tf.random_normal([n_hidden2]),name='b2'),
	'b_out_action': tf.Variable(tf.random_normal([n_output_action]),name='b_out_action'),
	'b_out_value': tf.Variable(tf.random_normal([n_output_value]),'b_out_value')
}

#Read the control parameters and variables
def initiliaze_nnet():
	global T
	global First_time
	global Maxtrajectory
	#Assume the first line indicate if it's the first time to run the code
	Load=np.loadtxt('control.txt')
	First_time=Load[0]
	T=Load[1]
	Maxtrajectory=Load[2]

	#Need to initialize the Tensorflow Variables LATER
	if First_time==1: 
		Load[0]=0
		Load[1]=1
		T=1
	#T represent the current trajectory number and 
	#we will update the parameters when T=1
	if T>=Maxtrajectory: 
		Load[1]=1
	else:
		Load[1]=Load[1]+1

	np.savetxt('control.txt', Load)

def lossfun():
	loss_R=0.0
	loss_V=0.0
	onestate=tf.slice(state_batch,[int(Maxtrajectory-1),0],[1,n_input])
	_,R=ff_nnet(onestate)
	for i in range(int(Maxtrajectory-1),0,-1):
		R=tf.add(reward_batch[i-1],tf.multiply(R,Dicount_factor))
		onestate=tf.slice(state_batch,[i-1,0],[1,n_input])
		Paction,Pvalue=ff_nnet(onestate)
		index=tf.cast(action_batch[i-1,0],tf.int32)

		loss_R=tf.add(loss_R,tf.multiply(tf.log(Paction[0,index]),tf.subtract(R,Pvalue)))
		loss_V=tf.add(loss_V,tf.square(tf.subtract(R,Pvalue)))
	return loss_V,-loss_R


#Define the computation graph
actions,value=ff_nnet(x)
loss_V,loss_R=lossfun()
train_step1 = tf.train.AdamOptimizer(1e-4).minimize(loss_R)
train_step2 = tf.train.AdamOptimizer(1e-4).minimize(loss_V)
init = tf.global_variables_initializer()
saver = tf.train.Saver()


def main():
	#Initializing the neural net
	initiliaze_nnet()

	with tf.Session() as sess:
		if First_time==1:
			sess.run(init)
			print('Model initialiazed')
			saver.save(sess, "./kdnnet.ckpt")
			print('Modeled saved\n')
			# Initialize the state.txt action.txt obj_value.csv
			with open('state.txt', 'w') as f:
				f.truncate()
			with open('action.txt', 'w') as f:
				f.truncate()
			with open('obj_value.txt', 'w') as f:
				f.truncate()

		else:
			saver.restore(sess, "./kdnnet.ckpt")
			print("Model restored.\n")


		#Train the network when a new batch starts, Save the network variaables, Clear the state, action and rewards file
		if (T==1)and(First_time!=1):
			#Load the aciton.txt and state.txt for training
			train_states,train_actions,train_rewrads=load_file()
			#Train the neural network
			train_step1.run(feed_dict={action_batch:train_actions,state_batch:train_states,reward_batch:train_rewrads})
			train_step2.run(feed_dict={action_batch: train_actions, state_batch: train_states, reward_batch: train_rewrads})
			saver.save(sess, "./kdnnet.ckpt")
			print('Training process completed')
		#Make an action with the current policy
		#Action=1: do the maximum match, =0: do nothing
		P_actions=sess.run(actions,feed_dict={x:initial_state})
		Action= [0] if np.random.uniform()<P_actions[0,0] else [1]

		#Save the current state and action into state.txt and action.txt
		with open('state.txt','a') as st:
			np.savetxt(st,initial_state)
			st.close()
		with open('action.txt','a') as act:
			np.savetxt(act,Action)
			act.close()

def load_file():
	#Load states and actions
	#Note:each row represent one data
	states=np.loadtxt('state.txt')
	states=states.reshape(int(Maxtrajectory),n_input)

	actions=np.loadtxt('action.txt')
	actions=actions.reshape(int(Maxtrajectory),1)

	rewards=np.loadtxt('obj_value.txt')
	rewards=rewards[0:int(Maxtrajectory)]
	rewards=rewards.reshape(int(Maxtrajectory),1)

	#Clear the states and actions files for the new iteration
	with open('state.txt','w') as f:
		f.truncate()
	with open('action.txt','w') as f:
		f.truncate()
	with open('obj_value.txt','w') as f:
		f.truncate()

	return states,actions,rewards

main()

# Deep-Kidney-Exchange
The repository is the implementation of deep rereinforcement learning on Kidney Exchange problem based on Tensorflow. The design and implementation is based on the JAVA codebase [Kidney Exchange](https://github.com/JohnDickerson/KidneyExchange).
## Asynchronous advantage actor critic method (A3C)
The file '*AsynAC_kd.py*' is the implementation. Everytime the file '*AsynAC_kd.py*' is called, it will read **one single state** from '*st_1.csv*' and generate **one action** based on the current neural net. To run the file, it needs '*st_1.csv*', '*control.txt*' and '*obj_value.txt*' provided from the code base. In the folder, manually created examples are included. The arguments are as follows:
```
Input Arguments:
	st_1.csv: file that save the current state
	control_parameters.txt: file that control the training and running process
		1st line:'0' load the neural net tf.Variables from the file kdnnet.ckpt
		          '1' first time run the code, and need to initialize tf.Variables

		2nd line: represent the current trajectory number T, when T>Maxtrajectory, train neural net
		3rd line: Maxtrajectory: the batch size of data used to train the neural net
	obj_value.txt: Batch of rewards provided by the Java part

Output Arguments:
	action.txt, state.txt: Batch of states and actions mantained by the python file itself, used to train the neural net
	kdnnet.ckpt: Save the tf.Variables of the neural net
  ```

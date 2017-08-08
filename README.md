# Deep-Kidney-Exchange
The repository is the implementation of deep rereinforcement learning algorithm on Kidney Exchange problem based on Tensorflow. The implementation is based on the OpenAI Gym environment. Each algorithm can be modified to work on the specified Kidney Exchange model. The Kidney Exchange simulator based on the OpenAI Gym can be found [here](https://github.com/camoy/gym-kidney)
## Dependencies
* Python3
* [OpenAI Gym](https://github.com/openai/gym)
* [Kidney Exchange environment](https://github.com/camoy/gym-kidney)
* [OpenAI baselines](https://github.com/openai/baselines)
* [OpenMPI](https://www.open-mpi.org/)
* [Mpi4y](https://pypi.python.org/pypi/mpi4py)
* [Spams](http://spams-devel.gforge.inria.fr/downloads.html)
* [Gurobi]()
* Tensorflow
* NumPy
* SciPy
* NetworkX
## How to run
For each algrithm, you need to specify the parameters of the corresponding model on the top of code, and then simply run the `test.py` file. For example, 
```
python3 test_trpo.py
```
Note: If you have a issue with the Mpi4y,you may have to set the environment variable `TMPDIR`as `/tmp`by:
```
export TMPDIR=/tmp
```
## Reference
* [Arthur Juliani's A3C Doom implementation](https://github.com/awjuliani/DeepRL-Agents/blob/master/A3C-Doom.ipynb)
* [OpenAI baselines](https://github.com/openai/baselines)

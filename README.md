# Deep-Kidney-Exchange
The repository implements the deep RL TRPO with LSTM on Kidney Exchange based on Tensorflow. The Kidney Exchange simulator based on the OpenAI Gym can be found [here](https://github.com/camoy/gym-kidney). A simplified dynamic set packing model can be found[here](https://github.com/Pengchengpcx/Dynamic-Set-Packing).
## Dependencies
* Python3
* [OpenAI Gym](https://github.com/openai/gym)
* [Kidney Exchange environment](https://github.com/camoy/gym-kidney)
* [OpenAI baselines](https://github.com/openai/baselines)
* [OpenMPI](https://www.open-mpi.org/)
* [Mpi4y](https://pypi.python.org/pypi/mpi4py)
* [Spams](http://spams-devel.gforge.inria.fr/downloads.html)
* [BLAS](http://www.netlib.org/blas/)
* [Gurobi](http://www.gurobi.com/downloads/gurobi-optimizer)
* Tensorflow
* NumPy
* SciPy
* NetworkX
## How to run
If you have a issue with the Mpi4y,you may have to set the environment variable `TMPDIR`as `/tmp`by:
```
export TMPDIR=/tmp
```
If you use multithreading program and OpenBLAS at the same time, you may have to change the environment variable of OpenBLAS multihreading since it may controdict with your original multithreading setting.

`export OPENBLAS_NUM_THREADS=1` in the environment variables. Or
Call `openblas_set_num_threads(1)` in the application on runtime. Or
Build OpenBLAS single thread version, e.g. `make USE_THREAD=0`

## Reference
* [OpenAI baselines](https://github.com/openai/baselines)

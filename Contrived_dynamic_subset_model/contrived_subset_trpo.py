from mpi4py import MPI
from baselines.common import set_global_seeds
import os.path as osp
import logging
from baselines import logger
from baselines.pposgd.mlp_policy import MlpPolicy
from baselines.common.mpi_fork import mpi_fork
from baselines import bench
from baselines.trpo_mpi import trpo_mpi
import trpo_indi
import spams
import sys
import gym
import envs
from envs.wrapper.ConfigWrapper import SubsetWrapper


MODEL_NAME="Contrived_subset-v0"


# Training parameters
hidden_layers=4

# Note: incorporate the k to the equation
hidden_units=200
maxepisodes=30000
timesteps_per_batch=5000 # Run current policy number of steps per iteration
max_kl=0.01
cg_iters=10
cd_damping=0.1
gamma=0.99
lam=0.98
vf_iters=5
vf_stepsize=1e-6
num_cpu=1
Vcoprecent = 0.7
theta = 0.6



def train(env_id, num_timesteps, seed, model_name, model_path, para,load_model,
          timesteps_per_batch,hidden_units,hidden_layers):
    whoami  = mpi_fork(num_cpu)
    if whoami == "parent":
        return
    import baselines.common.tf_util as U
    logger.session().__enter__()
    sess = U.single_threaded_session()
    sess.__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env = gym.make(env_id)
    env = SubsetWrapper(env, para)
    #env = gym_kidney.LogWrapper(env, NN, EXP, OUT, FREQ, PARAM)
    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name,
                         ob_space=env.observation_space,
                         ac_space=env.action_space,
                         hid_size=hidden_units,
                         num_hid_layers=hidden_layers)
    env = bench.Monitor(env, osp.join(logger.get_dir(), "%i.monitor.json" % rank))
    # env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    trpo_indi.learn(env, policy_fn,
                   timesteps_per_batch=timesteps_per_batch,
                   max_kl=max_kl, cg_iters=cg_iters,
                   cg_damping=cd_damping,
                   max_episodes=num_timesteps,
                   gamma=gamma, lam=lam,
                   vf_iters=vf_iters,
                   vf_stepsize=vf_stepsize,
                   load_model=load_model,
                   model_path=model_path
                    )
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--p', help='Compatible Vertices Distribution', type=float, default=0.9)
    parser.add_argument('--theta', help='Reset Time', type=float, default=0.3)
    parser.add_argument('--N', help='Total Coming Vertices', type=int, default=300)
    parser.add_argument('--loadmodel', help='Load the Neural Net', type=bool, default=False)
    parser.add_argument('--record', help='Record the Policy performance', type= bool,default=False)
    parser.add_argument('--trainsize', help='Training trajecotries', type=int, default=20000)
    parser.add_argument('--batchsize', help='Batch time steps in each update', type=int, default=6000)
    parser.add_argument('--hiddenunit', help='Hidden units for each layer in Neural Net', type=int, default=200)
    parser.add_argument('--hiddenlayers', help='Hidden layers for each layer in Neural Net', type=int, default=4)
    args = parser.parse_args()

    env_id = "Contrived_subset-v0"
    maxepisodes = args.trainsize
    timesteps_per_batch = args.batchsize
    hidden_units = args.hiddenunit
    hidden_layers = args.hiddenlayers

    # for i in range(2):
    p = args.p
    theta = args.theta
    N = args.N
    model_path = './NN/p%stheta%smean%s/' %(p,theta,N)
    load_model = args.loadmodel
    para = {
        'p': p,
        'mean': N,
        'theta': theta,
        'record': args.record,
        'path': './RL/p%stheta%smean%s_data.csv' % (p,theta, N)
    }

    train(env_id=env_id, num_timesteps=maxepisodes, seed=0, model_name=MODEL_NAME, model_path=model_path,
               para= para,load_model=load_model,timesteps_per_batch=timesteps_per_batch,
          hidden_layers=hidden_layers, hidden_units=hidden_units
          )

if __name__ == '__main__':
    main()

from mpi4py import MPI
from baselines.common import set_global_seeds
import os.path as osp
import gym
import gym_kidney
import logging
from baselines import logger
from baselines.pposgd.mlp_policy import MlpPolicy
from baselines.common.mpi_fork import mpi_fork
from baselines import bench
from baselines.trpo_mpi import trpo_mpi
import trpo_indi
import sys

# Environment parameters
env_id="kidney-v0"
homogeneous={
    'tau':7,
    'rate':9,
    'k':3,
    'p':0.1,
    'p_a':0.01
}
heterogeneous={
    'tau':7,
    'rate':25,
    'k':50,
    'p':0.05,
    'p_l':0.1,
    'p_h':0.1,
    'p_a':0.1,
}
kidney={
    'seed':2618,
    'tau':5,
    'alpha':0.05,
    't':3,
    'cycle_cap':3,
    'chain_cap':3,
    'k':24,
    'm':580,
    "data": "/Users/PengchengXu/deep36/test_ppo/data_adj.csv",
    "details": "/Users/PengchengXu/deep36/test_ppo/data_details.csv"
}
contrived={
    'tau': 6
}
MODEL=contrived
MODEL_NAME="contrived"

# Training parameters
hidden_layers=4

# Note: incorporate the k to the equation
hidden_units=(MODEL['tau']**2+MODEL['tau'])*4
maxepisodes=1000
timesteps_per_batch=150 # Run current policy number of steps per iteration
max_kl=0.01
cg_iters=10
cd_damping=0.1
maxepisdoes=1000
gamma=0.99
lam=0.98
vf_iters=5
vf_stepsize=1e-5
num_cpu=1

model_path='./model/trpo/'+MODEL_NAME
load_model=True



def train(env_id, num_timesteps, seed, model_name, model_type):
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
    env = gym_kidney.ConfigWrapper(env, model_name, model_type)
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
    train(env_id=env_id, num_timesteps=maxepisdoes, seed=0, model_name=MODEL_NAME, model_type=MODEL)

if __name__ == '__main__':
    main()

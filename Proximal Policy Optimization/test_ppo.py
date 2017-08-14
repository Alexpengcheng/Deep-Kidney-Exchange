from baselines.common import set_global_seeds, tf_util as U
import pposgd_indi
from baselines import bench
import os.path as osp
import gym, logging
from baselines import logger
import gym_kidney

# Environment parameters
env_id="kidney-v0"
EXP = 0
OUT = "/home/alexxu/deepkidney/Deep-Kidney-Exchange/Proximal Policy Optimization/pposummary"
FREQ = 1
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
    'tau':7,
    'alpha':0.05,
    't':5,
    'cycle_cap':3,
    'chain_cap':3,
    'k':120,
    'm':30,
    "data": "/home/alexxu/data_adj.csv",
	"details": "/home/alexxu/data_details.csv",
    "d_path":"/home/alexxu/dictionary.tar.gz"
}
contrived={
    'tau': 7
}
MODEL=contrived
MODEL_NAME="contrived"

# Training parameters
hidden_layers=4

# Note: incorporate the k to the equation
hidden_units=(MODEL['tau']**2+MODEL['tau'])*4
maxepisodes=1000
timesteps_per_batch=150 # Run current policy number of steps per iteration
clip_param=0.2
entropy_coeff=0.01
optim_epochs=10
optim_stepsize=1e-5
optim_batchsize=50 # should be smaller than timesteps_per_batch
gamma=0.99
lam=0.95

model_path="/home/alexxu/deepkidney/Deep-Kidney-Exchange/Proximal Policy Optimization/model"+MODEL_NAME
load_model=True


def train(env_id, num_timesteps, seed, model_name, model_type):
    from baselines.pposgd import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    logger.session().__enter__()
    set_global_seeds(seed)
    env = gym.make(env_id)
    env = gym_kidney.ConfigWrapper(env, model_name, model_type)

    # Define neural net policy
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=hidden_units, num_hid_layers=hidden_layers)

    # Define the environment
    env = bench.Monitor(env, osp.join(logger.get_dir(), "monitor.json"))
    gym.logger.setLevel(logging.WARN)

    # Define the environment
    pposgd_indi.learn(env, policy_fn,
                        max_episodes=num_timesteps,
                        timesteps_per_batch=timesteps_per_batch,
                        clip_param=clip_param, entcoeff=entropy_coeff,
                        optim_epochs=optim_epochs, optim_stepsize=optim_stepsize, optim_batchsize=optim_batchsize,
                        gamma=gamma, lam=lam,
                        load_model=load_model,
                        model_path=model_path
                        )
    env.close()

def main():
    train(env_id=env_id, num_timesteps=maxepisodes, seed=0, model_name=MODEL_NAME, model_type=MODEL)


if __name__ == '__main__':
    main()





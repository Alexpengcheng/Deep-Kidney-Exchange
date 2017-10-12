from mpi4py import MPI
from baselines.common import set_global_seeds
import os.path as osp
import os
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
import gym_kidney
from gym_kidney import actions, embeddings, \
	models, loggers, wrappers




# Training parameters

# Note: incorporate the k to the equation
max_kl=0.01
cg_iters=10
cd_damping=0.1
gamma=0.99
lam=0.98
vf_iters=5
vf_stepsize=1e-6
num_cpu=1



def train(env_id, num_timesteps, seed, model_path, load_model,
          timesteps_per_batch,hidden_units,hidden_layers,
          ACTION, EMBEDDING, MODEL, LOGGING):
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

    # workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    workerseed = 2221438774
    set_global_seeds(workerseed)
    env = gym.make(env_id)
    env = wrappers.ConfigWrapper(env, ACTION, EMBEDDING, MODEL, LOGGING)
    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name,
                         ob_space=env.observation_space,
                         ac_space=env.action_space,
                         hid_size=hidden_units,
                         num_hid_layers=hidden_layers)
    env = bench.Monitor(env, osp.join(logger.get_dir(), "%i.monitor.json" % rank))
    env.seed(workerseed)
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
    '''
    All Input Parameters
    '''
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--loadmodel', help='Load the Neural Net', type=bool, default=False)
    parser.add_argument('--trainsize', help='Training trajecotries', type=int, default=20000)
    parser.add_argument('--batchsize', help='Batch time steps in each update', type=int, default=20000)
    parser.add_argument('--hiddenunit', help='Hidden units for each layer in Neural Net', type=int, default=200)
    parser.add_argument('--hiddenlayers', help='Hidden layers for each layer in Neural Net', type=int, default=4)

    parser.add_argument('--cyclecap', help='cyclecap', type=int, default=2)
    parser.add_argument('--chaincap', help='chaincap', type=int, default=2)
    parser.add_argument('--action', help='action space', type=str, default='flap')
    parser.add_argument('--M', help='Cardinality', type=int, default=32)
    parser.add_argument('--K', help='Match frequency', type=int, default=32)
    parser.add_argument('--LEN', help='length ', type=int, default=2000)

    parser.add_argument('--sample', help='sample length ', type=int, default=5)
    parser.add_argument('--chainsample', help='sample cap of chain ', type=int, default=2)
    parser.add_argument('--cyclesample', help='sample cap of cycle ', type=int, default=2)
    parser.add_argument('--tau', help='walk2vec:tau', type=float, default=7)
    parser.add_argument('--alpha', help='walk2vec:alpha', type=float, default=0.05)


    args = parser.parse_args()

    '''
    Environment Parameters Setting
    '''
    env_id = "kidney-v0"

    # Local constants
    SHOW = False

    # Action constants
    CYCLE_CAP = args.cyclecap
    CHAIN_CAP = args.chaincap
    action = args.action
    if action == 'flap':
        ACTION = actions.FlapAction(CYCLE_CAP, CHAIN_CAP)
    elif action == 'blood':
        ACTION = actions.BloodAction(CYCLE_CAP, CHAIN_CAP)
    else:
        raise NotImplementedError

    # Embedding constants
    SAMPLE_LENGTH = args.sample
    CHAIN_SAMPLE = args.chainsample
    CYCLE_SAMPLE = args.cyclesample
    TAU = args.tau
    ALPHA = args.alpha
    all_embedding = [embeddings.ChainEmbedding(CHAIN_SAMPLE),embeddings.CycleFixedEmbedding(SAMPLE_LENGTH,CYCLE_SAMPLE),embeddings.CycleVariableEmbedding(1,SAMPLE_LENGTH,CYCLE_SAMPLE),
                     embeddings.DdEmbedding(),embeddings.NddEmbedding(),embeddings.NopEmbedding(),embeddings.OrderEmbedding(),
                     embeddings.Walk2VecEmbedding([embeddings.p0_max, embeddings.p0_mean],TAU,ALPHA)]

    embed = []
    scale = []
    normal = [0,0,0,0, 0,0,1,0]
    for i in range(len(all_embedding)):
        if normal[i] != 0:
            embed.append(all_embedding[i])
            scale.append(normal[i])
    EMBEDDING = embeddings.UnionEmbedding(embed)
    EMBEDDING = embeddings.NormalizeEmbedding(embeddings.UnionEmbedding(embed), scale)

    # Model constants
    M = args.M
    K = args.K
    LEN = args.LEN
    DET_PATH = './data_details.csv'
    ADJ_PATH = './data_adj.csv'
    MODEL = models.DataModel(M, K, ADJ_PATH, DET_PATH, LEN)

    # Logging constants
    os.system('mkdir ./record/M%sK%sCycle%sChain%sAction%sEmbed'% (M, K, CYCLE_CAP, CHAIN_CAP,action))
    PATH = './record/M%sK%sCycle%sChain%sAction%sEmbed/' % (M, K, CYCLE_CAP, CHAIN_CAP,action)
    EXP = 0
    CUSTOM = {"agent": "rl_trpo"}
    LOGGING = loggers.CsvLogger(PATH, EXP, CUSTOM)



    '''
    Training Parameters
    '''
    maxepisodes = args.trainsize
    timesteps_per_batch = args.batchsize
    hidden_units = args.hiddenunit
    hidden_layers = args.hiddenlayers


    model_path = './NN/M%sK%sCycle%sChain%sAction%s/' % (M, K, CYCLE_CAP, CHAIN_CAP,action)
    load_model = args.loadmodel


    train(env_id=env_id, num_timesteps=maxepisodes, seed=0,  model_path=model_path,
          load_model=load_model,timesteps_per_batch=timesteps_per_batch,
          hidden_layers=hidden_layers, hidden_units=hidden_units,
          ACTION=ACTION, EMBEDDING=EMBEDDING, MODEL=MODEL, LOGGING=LOGGING
          )

if __name__ == '__main__':
    main()
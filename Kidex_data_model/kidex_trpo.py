from mpi4py import MPI
from baselines.common import set_global_seeds
import os.path as osp
import os
import logging
from baselines import logger
from trpo_policies import LSTMPolicy, MlpPolicy
from baselines.common.mpi_fork import mpi_fork
from baselines import bench
from baselines.trpo_mpi import trpo_mpi
import trpo_indi, trpo_indi_mlp
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
lam=0.97
vf_iters=10
vf_stepsize=1e-7
num_cpu=1

def helper(var):
    if var == 'True':
        return True
    elif var == 'False':
        return False
    else:
        raise NotImplementedError

def train(env_id, num_timesteps, seed, model_path, load_model,
          timesteps_per_batch,hidden_units,hidden_layers, trainmodel,
          ACTION, EMBEDDING, MODEL, LOGGING):
    whoami  = mpi_fork(num_cpu)
    if whoami == "parent":
        return
    import baselines.common.tf_util as U
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
        return LSTMPolicy(name=name,
                         ob_space=ob_space,
                         ac_space=ac_space,
                         hid_size=hidden_units,
                         num_hid_layers=hidden_layers)
    env.seed(workerseed)

    trpo_indi.learn(env, policy_fn,
                   timesteps_per_batch=timesteps_per_batch,
                   max_kl=max_kl, cg_iters=cg_iters,
                   cg_damping=cd_damping,
                   max_episodes=num_timesteps,
                   gamma=gamma, lam=lam,
                   vf_iters=vf_iters,
                   vf_stepsize=vf_stepsize,
                   load_model=load_model,
                   model_path=model_path,
                    trainmodel= trainmodel
                    )
    env.close()

def main():
    '''
    All Input Parameters
    '''
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--loadmodel', help='Load the Neural Net', type=str, default='False')
    parser.add_argument('--trainsize', help='Training trajecotries', type=int, default=200000)
    parser.add_argument('--batchsize', help='Batch time steps in each update', type=int, default=8000)
    parser.add_argument('--hiddenunit', help='Hidden units for each layer in Neural Net', type=int, default=64)
    parser.add_argument('--hiddenlayers', help='Hidden layers for each layer in Neural Net', type=int, default=3)

    parser.add_argument('--cyclecap', help='cyclecap', type=int, default=2)
    parser.add_argument('--chaincap', help='chaincap', type=int, default=2)
    parser.add_argument('--action', help='action space', type=str, default='blood')
    parser.add_argument('--M', help='Cardinality', type=int, default=256)
    parser.add_argument('--K', help='Match frequency', type=int, default=512)
    parser.add_argument('--LEN', help='length ', type=int, default=3000)

    parser.add_argument('--sample', help='sample length ', type=int, default=5)
    parser.add_argument('--chainsample', help='sample cap of chain ', type=int, default=2)
    parser.add_argument('--cyclesample', help='sample cap of cycle ', type=int, default=2)
    parser.add_argument('--tau', help='walk2vec:tau', type=int, default=7)
    parser.add_argument('--alpha', help='walk2vec:alpha', type=float, default=0.05)
    parser.add_argument('--time', help='sojourn', type=float, default=20)
    parser.add_argument('--trainmodel', help='if train the model', type=str, default='True')



    args = parser.parse_args()

    '''
    Environment Parameters Setting
    '''
    env_id = "kidney-v0"

    # Local constants
    SHOW = False

    # Action constants
    W_FUN = lambda o1, o2: 1 - 0.5 * (o1 + o2)
    CYCLE_CAP = args.cyclecap
    CHAIN_CAP = args.chaincap
    action = args.action
    if action == 'flap':
        ACTION = actions.FlapAction(CYCLE_CAP, CHAIN_CAP)
    elif action == 'blood':
        ACTION = actions.BloodAction(CYCLE_CAP, CHAIN_CAP, -4, 4, W_FUN)
    else:
        raise NotImplementedError

    # Embedding constants
    SAMPLE_LENGTH = args.sample
    CHAIN_SAMPLE = args.chainsample
    CYCLE_SAMPLE = args.cyclesample
    TAU = args.tau
    ALPHA = args.alpha
    trainmodel = helper(args.trainmodel)
    all_embedding = [embeddings.ChainEmbedding(CHAIN_SAMPLE),embeddings.CycleFixedEmbedding(SAMPLE_LENGTH,CYCLE_SAMPLE),embeddings.CycleVariableEmbedding(1,SAMPLE_LENGTH,CYCLE_SAMPLE),
                     embeddings.DdEmbedding(),embeddings.NddEmbedding(),embeddings.NopEmbedding(),embeddings.OrderEmbedding(),
                     embeddings.Walk2VecEmbedding([embeddings.p0_max, embeddings.p0_mean],TAU,ALPHA)]

    embed = []
    scale = []
    normal = [0.05, 0.05, 0.02, 0.02, 0.02, 0, 0.02 , 0]
    for i in range(len(all_embedding)):
        if normal[i] != 0:
            embed.append(all_embedding[i])
            if i == 7: # for walk2vec embedding
                scale += [normal[i]]* int((TAU**2 +TAU))
            else:
                scale.append(normal[i])
    EMBEDDING = embeddings.NormalizeEmbedding(embeddings.UnionEmbedding(embed), scale)

    # Model constants
    M = args.M
    K = args.K
    LEN = 9*K
    TIME = args.time
    DET_PATH = './data_details.csv'
    ADJ_PATH = './data_adj.csv'
    MODEL = models.DataModel(M, K, ADJ_PATH, DET_PATH, LEN, TIME)

    # Logging constants
    os.system('mkdir ./record/LSTMrlM%sK%sCycle%sChain%sAction%sEmbedTime%s'% (M, K, CYCLE_CAP, CHAIN_CAP,action, TIME))
    PATH = './record/LSTMrlM%sK%sCycle%sChain%sAction%sEmbedTime%s/' % (M, K, CYCLE_CAP, CHAIN_CAP,action, TIME)
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
    load_model = helper(args.loadmodel)


    train(env_id=env_id, num_timesteps=maxepisodes, seed=0,  model_path=model_path,
          load_model=load_model,timesteps_per_batch=timesteps_per_batch,
          hidden_layers=hidden_layers, hidden_units=hidden_units,
          ACTION=ACTION, EMBEDDING=EMBEDDING, MODEL=MODEL, LOGGING=LOGGING, trainmodel=trainmodel
          )

if __name__ == '__main__':
    main()
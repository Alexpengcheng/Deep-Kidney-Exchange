#!/usr/bin/env python
import argparse
import logging
import os
import tensorflow as tf
import gym
from baselines import logger
from baselines.common import set_global_seeds
from baselines import bench
from acktr_cont_indi import learn
from acktr_policies import GaussianMlpPolicy
from acktr_valuefunction import NeuralNetValueFunction
import gym_kidney
from gym_kidney import actions, embeddings, \
	models, loggers, wrappers

def train(env_id, num_timesteps, seed, timesteps_per_batch, hidden_layers, hidden_units,
          ACTION, EMBEDDING, MODEL, LOGGING, model_path, load_model):

    env = gym.make(env_id)
    print ('env.spec.timestep_limit:',env.spec.timestep_limit)
    env = wrappers.ConfigWrapper(env, ACTION, EMBEDDING, MODEL, LOGGING)
    if logger.get_dir():
        env = bench.Monitor(env, os.path.join(logger.get_dir(), "monitor.json"))
    set_global_seeds(seed)
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)

    with tf.Session(config=tf.ConfigProto()) as session:
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        with tf.variable_scope("vf"):
            vf = NeuralNetValueFunction(ob_dim, ac_dim, hidden_layers, hidden_units)
        with tf.variable_scope("pi"):
            policy = GaussianMlpPolicy(ob_dim, ac_dim, hidden_layers, hidden_units)

        learn(load_model=load_model, model_path=model_path, env=env, policy=policy, vf=vf,
            gamma=0.99, lam=0.97, timesteps_per_batch=timesteps_per_batch,
            desired_kl=0.002,
            num_timesteps=num_timesteps, animate=False)

        env.close()

def main():
    '''
    All Input Parameters
    '''
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--loadmodel', help='Load the Neural Net', type=bool, default=False)
    parser.add_argument('--trainsize', help='Training trajecotries', type=int, default=1e6)
    parser.add_argument('--batchsize', help='Batch time steps in each update', type=int, default=1000)
    parser.add_argument('--hiddenunit', help='Hidden units for each layer in Neural Net', type=int, default=64)
    parser.add_argument('--hiddenlayers', help='Hidden layers for each layer in Neural Net', type=int, default=2)

    parser.add_argument('--cyclecap', help='cyclecap', type=int, default=2)
    parser.add_argument('--chaincap', help='chaincap', type=int, default=2)
    parser.add_argument('--action', help='action space', type=str, default='blood')
    parser.add_argument('--M', help='Cardinality', type=int, default=256)
    parser.add_argument('--K', help='Match frequency', type=int, default=512)
    parser.add_argument('--LEN', help='length ', type=int, default=2000)

    parser.add_argument('--sample', help='sample length ', type=int, default=9)
    parser.add_argument('--chainsample', help='sample cap of chain ', type=int, default=6)
    parser.add_argument('--cyclesample', help='sample cap of cycle ', type=int, default=4)
    parser.add_argument('--tau', help='walk2vec:tau', type=float, default=5)
    parser.add_argument('--alpha', help='walk2vec:alpha', type=float, default=0.05)


    args = parser.parse_args()

    '''
    Environment Parameters Setting
    '''
    env_id = "MountainCarContinuous-v0"
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
    SAMPLE_LENGTH = CHAIN_CAP + CYCLE_CAP
    CHAIN_SAMPLE = CHAIN_CAP + 1
    CYCLE_SAMPLE = CYCLE_CAP + 1
    TAU = args.tau
    ALPHA = args.alpha
    all_embedding = [embeddings.ChainEmbedding(CHAIN_SAMPLE),embeddings.CycleFixedEmbedding(SAMPLE_LENGTH,CYCLE_SAMPLE),embeddings.CycleVariableEmbedding(1,SAMPLE_LENGTH,CYCLE_SAMPLE),
                     embeddings.DdEmbedding(),embeddings.NddEmbedding(),embeddings.NopEmbedding(),embeddings.OrderEmbedding(),
                     embeddings.Walk2VecEmbedding([embeddings.p0_max, embeddings.p0_mean],TAU,ALPHA)]

    embed = []
    scale = []
    normal = [0.05,0.005,0.002,0.004, 0.02,0,0.004,0]

    for i in range(len(all_embedding)):
        if normal[i] != 0:
            embed.append(all_embedding[i])
            scale.append(normal[i])

    EMBEDDING = embeddings.NormalizeEmbedding(embeddings.UnionEmbedding(embed), scale)

    # Model constants
    M = args.M
    K = args.K
    LEN = 3*K
    DET_PATH = './data_details.csv'
    ADJ_PATH = './data_adj.csv'
    MODEL = models.SparseModel(M, K, 0.7, ADJ_PATH, DET_PATH, LEN)

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


    train(env_id=env_id, num_timesteps=maxepisodes, seed=10,  model_path=model_path,
          load_model=load_model,timesteps_per_batch=timesteps_per_batch,
          hidden_layers=hidden_layers, hidden_units=hidden_units,
          ACTION=ACTION, EMBEDDING=EMBEDDING, MODEL=MODEL, LOGGING=LOGGING
          )

if __name__ == "__main__":
   main()

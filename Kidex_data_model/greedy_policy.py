import os
import logging
import sys
import gym
import gym_kidney
from gym_kidney import actions, embeddings, \
	models, loggers, wrappers



def main():
    '''
    All Input Parameters
    '''
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--trainsize', help='Training trajecotries', type=int, default=40000)

    parser.add_argument('--cyclecap', help='cyclecap', type=int, default=2)
    parser.add_argument('--chaincap', help='chaincap', type=int, default=2)
    parser.add_argument('--action', help='action space', type=str, default='blood')
    parser.add_argument('--M', help='Cardinality', type=int, default=256)
    parser.add_argument('--K', help='Match frequency', type=int, default=512)
    parser.add_argument('--LEN', help='length ', type=int, default=3000)

    parser.add_argument('--sample', help='sample length ', type=int, default=5)
    parser.add_argument('--chainsample', help='sample cap of chain ', type=int, default=2)
    parser.add_argument('--cyclesample', help='sample cap of cycle ', type=int, default=2)
    parser.add_argument('--tau', help='walk2vec:tau', type=float, default=7)
    parser.add_argument('--alpha', help='walk2vec:alpha', type=float, default=0.05)
    parser.add_argument('--time', help='sojourn', type=float, default=20)



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
    all_embedding = [embeddings.ChainEmbedding(CHAIN_SAMPLE),embeddings.CycleFixedEmbedding(SAMPLE_LENGTH,CYCLE_SAMPLE),embeddings.CycleVariableEmbedding(1,SAMPLE_LENGTH,CYCLE_SAMPLE),
                     embeddings.DdEmbedding(),embeddings.NddEmbedding(),embeddings.NopEmbedding(),embeddings.OrderEmbedding(),
                     embeddings.Walk2VecEmbedding([embeddings.p0_max, embeddings.p0_mean],TAU,ALPHA)]

    embed = []
    scale = []
    normal = [0.05,0.05,0.02,0.02, 0.02,0,0.02,0]
    for i in range(len(all_embedding)):
        if normal[i] != 0:
            embed.append(all_embedding[i])
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
    os.system('mkdir ./record/greM%sK%sCycle%sChain%sAction%sEmbedTime%s'% (M, K, CYCLE_CAP, CHAIN_CAP,action, TIME))
    PATH = './record/greM%sK%sCycle%sChain%sAction%sEmbedTime%s/' % (M, K, CYCLE_CAP, CHAIN_CAP,action, TIME)
    EXP = 0
    CUSTOM = {"agent": "rl_trpo"}
    LOGGING = loggers.CsvLogger(PATH, EXP, CUSTOM)

    '''
    Training Parameters
    '''
    maxepisodes = args.trainsize

    env = gym.make(env_id)
    env = wrappers.ConfigWrapper(env, ACTION, EMBEDDING, MODEL, LOGGING)

    for i in range(maxepisodes):
        obs, done = env.reset(), False
        while not done:
            if SHOW:
                env.render()
            obs, reward, done, _ = env.step(1)



if __name__ == '__main__':
    main()
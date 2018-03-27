import os
import sys
from csv import DictReader
from ftplib import FTP
import paramiko
import time
import csv

M = [64, 64, 64, 128, 128, 128, 256, 256, 256]
K = [32, 32, 32, 32, 32, 32, 32, 32, 32]
# t = [20.0, 40.0, 60.0, 20.0, 40.0, 60.0, 20.0, 40.0, 60.0]
t = [100.0, 150.0, 200.0, 600.0, 800.0, 1000.0, 600.0, 800.0, 1000.0]

def copytoVM():
    for i in range(12):
        os.system('scp -r ./kid_data_model alexxu@deepkidney'+str(i)+'.cs.umd.edu:/home/alexxu/deepkidney')
        # os.system('scp -r /Users/PengchengXu/Deep36/gym-kidney alexxu@deepkidney'+str(i)+'.cs.umd.edu:/home/alexxu/deepkidney')

class VMcontrol():
    def __init__(self, host , passwd, user = 'alexxu'):
        self.host = host
        self.user = user
        self.passwd = passwd
        self.server = paramiko.SSHClient()
        self.port = 22

    def execute(self,command):
        self.server.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.server.connect(hostname = self.host,port= self.port,username= self.user, password=self.passwd)
        self.server.exec_command(command)
        self.server.close()

def install_pack():
    for i in range(9,12):
        host = 'deepkidney' + str(i) + '.cs.umd.edu'
        user = 'alexxu'
        passwd = 'xpchXPCH1025'
        vm = VMcontrol(host=host, user=user, passwd=passwd)
        vm.execute('pkill -9 python')
        # vm.execute('cd deepkidney; rm -rf new_kidney')
        # vm.execute('cd deepkidney/set_packing; pip install -e . --user')

def delete_files():
    for i in range(9,12):
        host = 'deepkidney' + str(i) + '.cs.umd.edu'
        user = 'alexxu'
        passwd = 'xpchXPCH1025'
        vm = VMcontrol(host=host, user=user, passwd=passwd)
        vm.execute('cd /home/alexxu/deepkidney/kid_data_model/ ; '
                   'rm -rf greedyM* ; '
                   'rm -rf randomM* ; '
                   'rm -rf rlM* ; ')
        vm.execute('cd /home/alexxu/deepkidney/kid_data_model/NN ; '
                   'rm -rf M*')
        vm.execute('cd /home/alexxu/deepkidney/kid_data_model/record ; '
                   'rm -rf *')


def train_rl(i,j, M=M, K=K, t=t):
    CYCLE_CAP = 3
    CHAIN_CAP = 3
    SAMPLE = 3
    CHAINSAMPLE = 4
    CYCLESAMPLE = 4
    TAU = 5
    ALPHA = 0.05
    layers = 5
    action = 'flap'
    tsize = 6400000
    bsize = 80000

    for i in range(i,j):
        host = 'deepkidney' + str(i) + '.cs.umd.edu'
        user = 'alexxu'
        passwd = 'xpchXPCH1025'
        vm = VMcontrol(host=host, user=user, passwd=passwd)
        print('cd deepkidney/kid_data_model ;'
                   'nohup python3 kidex_trpo.py '
                   '--loadmodel %s '
                   '--trainmodel %s '
                   '--cyclecap %s '
                   '--chaincap %s '
                   '--trainsize %s '
                   '--batchsize %s '
                   '--M %s '
                   '--K %s '
                   '--hiddenlayers %s '
                   '--action %s '
                   '--sample %s '
                   '--chainsample %s '
                   '--cyclesampl %s '
                   '--tau %s '
                   '--alpha %s '
                   '--time %s '
                   '>rlM%sK%sChain%sCycle%s &' % ( 'False', 'True', CYCLE_CAP, CHAIN_CAP, tsize, bsize, M[i], K[i], layers, action, SAMPLE, CHAINSAMPLE, CYCLESAMPLE, TAU, ALPHA, t[i],
                                                M[i], K[i], CHAIN_CAP, CYCLE_CAP))
        time.sleep(1)

def train_greedy(i,j, M=M, K=K, t=t):
    CYCLE_CAP = 3
    CHAIN_CAP = 3
    SAMPLE = 9
    CHAINSAMPLE = 4
    CYCLESAMPLE = 4
    TAU = 5
    ALPHA = 0.05
    layers = 5
    action = 'flap'

    for i in range(i,j):
        host = 'deepkidney' + str(i) + '.cs.umd.edu'
        user = 'alexxu'
        passwd = 'xpchXPCH1025'
        vm = VMcontrol(host=host, user=user, passwd=passwd)
        print('cd deepkidney/kid_data_model ;'
                   'nohup python3 greedy_policy.py '
                   '--cyclecap %s '
                   '--chaincap %s '
                   '--M %s '
                   '--K %s '
                   '--action %s '
                   '--sample %s '
                   '--chainsample %s '
                   '--cyclesampl %s '
                   '--tau %s '
                   '--alpha %s '
                   '--time %s '
                   '>greedyM%sK%sChain%sCycle%sAction%s &' % (
                       CYCLE_CAP, CHAIN_CAP, M[i], K[i], action, SAMPLE, CHAINSAMPLE, CYCLESAMPLE, TAU, ALPHA,
                       t[i],
                       M[i], K[i], CHAIN_CAP, CYCLE_CAP, action))
        time.sleep(1)


def train_random(i,j, M=M, K=K, t=t):
    CYCLE_CAP = 3
    CHAIN_CAP = 3
    SAMPLE = 9
    CHAINSAMPLE = 4
    CYCLESAMPLE = 4
    TAU = 5
    ALPHA = 0.05
    action = 'flap'

    for i in range(i, j):
        host = 'deepkidney' + str(i) + '.cs.umd.edu'
        user = 'alexxu'
        passwd = 'xpchXPCH1025'
        vm = VMcontrol(host=host, user=user, passwd=passwd)
        print('cd deepkidney/kid_data_model ;'
                   'nohup python3 random_policy.py '
                   '--cyclecap %s '
                   '--chaincap %s '
                   '--M %s '
                   '--K %s '
                   '--action %s '
                   '--sample %s '
                   '--chainsample %s '
                   '--cyclesampl %s '
                   '--tau %s '
                   '--alpha %s '
                   '--time %s '
                   '>randomM%sK%sChain%sCycle%s &' % (
                       CYCLE_CAP, CHAIN_CAP, M[i], K[i], action, SAMPLE, CHAINSAMPLE, CYCLE_CAP, TAU, ALPHA,
                       t[i],
                       M[i], K[i], CHAIN_CAP, CYCLE_CAP))
        time.sleep(1)


def fetch_data(i,j):
    '''
    :param i: initial VM number inclusive
    :param j: last VM number exclusive
    :return: all statistics files
    '''
    for i in range (i,j):
        os.system('scp -r alexxu@deepkidney'+str(i)+'.cs.umd.edu:/home/alexxu/deepkidney/kid_data_model/record/ '
                                                    '/Users/PengchengXu/deep36/iclr_kidex/kidney2/datamodel_stat')

def statistics(M=M, K=K, TIME=t):
    CYCLE_CAP = 3
    CHAIN_CAP = 3
    action = 'flap'
    action2 = 'blood'

    for i in range(len(M)):
        rlpath = './datamodel_stat/record/LSTMrlM%sK%sCycle%sChain%sAction%sEmbedTime%s/000_STAT.csv' % (M[i], K[i], CYCLE_CAP, CHAIN_CAP, action, TIME[i])
        rlpath2 = './datamodel_stat/record/LSTMrlM%sK%sCycle%sChain%sAction%sEmbedTime%s/000_STAT.csv' % (M[i], K[i], CYCLE_CAP, CHAIN_CAP, action2, TIME[i])
        grepath = './datamodel_stat/record/greM%sK%sCycle%sChain%sAction%sEmbedTime%s/000_STAT.csv' % (M[i], K[i], CYCLE_CAP, CHAIN_CAP,action, TIME[i])
        randpath = './datamodel_stat/record/randM%sK%sCycle%sChain%sAction%sEmbedTime%s/000_STAT.csv' % (M[i], K[i], CYCLE_CAP, CHAIN_CAP, action, TIME[i])

        with open(grepath) as gre:
            reader = csv.DictReader(gre)
            gre_cycle_reward = []
            gre_chain_reward = []
            gre_arrived = []
            gre_departed = []
            for row in reader:
                gre_cycle_reward.append(int(row['cycle_reward']))
                gre_chain_reward.append(int(row['chain_reward']))
                gre_arrived.append(int(row['arrived']))
                gre_departed.append(int(row['departed']))

        with open(rlpath) as rl:
            reader = csv.DictReader(rl)
            rl_cycle_reward = []
            rl_chain_reward = []
            rl_departed = []
            rl_arrived = []
            for row in reader:
                rl_cycle_reward.append(int(row['cycle_reward']))
                rl_chain_reward.append(int(row['chain_reward']))
                rl_departed.append(int(row['departed']))
                rl_arrived.append(int(row['arrived']))

        with open(rlpath2) as rl:
            reader = csv.DictReader(rl)
            rl_cycle_reward2 = []
            rl_chain_reward2 = []
            rl_departed2 = []
            rl_arrived2 = []
            for row in reader:
                rl_cycle_reward2.append(int(row['cycle_reward']))
                rl_chain_reward2.append(int(row['chain_reward']))
                rl_departed2.append(int(row['departed']))
                rl_arrived2.append(int(row['arrived']))

        with open(randpath) as rand:
            reader = csv.DictReader(rand)
            rand_cycle_reward = []
            rand_chain_reward = []
            rand_arrived = []
            rand_departed = []
            for row in reader:
                rand_cycle_reward.append(int(row['cycle_reward']))
                rand_chain_reward.append(int(row['chain_reward']))
                rand_arrived.append(int(row['arrived']))
                rand_departed.append(int(row['departed']))

        print('*********************Results M%s K%s Time%s*************************' %(M[i], K[i], TIME[i]))
        ave1 = sum(gre_chain_reward)/(len(gre_chain_reward))
        ave2 = sum(gre_cycle_reward)/(len(gre_cycle_reward))
        ave11 = sum(gre_arrived)/(len(gre_arrived))
        ave12 = sum(gre_departed)/(len(gre_departed))

        ave3 = sum(rand_chain_reward)/(len(rand_chain_reward))
        ave4 = sum(rand_cycle_reward)/(len(rand_cycle_reward))
        ave31 = sum(rand_arrived)/(len(rand_arrived))
        ave32 = sum(rand_departed)/(len(rand_departed))


        ave5 = sum(rl_chain_reward[-800:])/(len(rl_chain_reward[-800:]))
        ave6 = sum(rl_cycle_reward[-800:])/(len(rl_cycle_reward[-800:]))
        ave7 = sum(rl_departed[-800:])/(len(rl_departed[-800:]))
        ave8 = sum(rl_arrived[-800:])/(len(rl_arrived[-800:]))

        ave52 = sum(rl_chain_reward2[-1000:]) / (len(rl_chain_reward2[-1000:]))
        ave62 = sum(rl_cycle_reward2[-1000:]) / (len(rl_cycle_reward2[-1000:]) )
        ave72 = sum(rl_departed2[-1000:]) / (len(rl_departed2[-1000:]) )
        ave82 = sum(rl_arrived2[-1000:]) / (len(rl_arrived2[-1000:]))


        print('Greedy Results: arrived %s, departed %s, chain %s, cycle %s, reward %s' %(ave11, ave12, ave1, ave2, ave1+ave2) )
        print('Random Results: arrived %s, departed %s, chain %s, cycle %s, reward %s' %(ave31, ave32, ave3, ave4, ave3+ave4) )
        print('Rl flap Results: arrived %s, departed %s, chain %s, cycle %s, reward %s' %(ave8, ave7, ave5, ave6, ave5+ave6) )
        print('Rl blood Results: arrived %s, departed %s, chain %s, cycle %s, reward %s' %(ave82, ave72, ave52, ave62, ave52+ave62) )


        print('\n')

#
# install_pack()
# delete_files()

# for i in range(4):
#     print(i)
#     train_rl(i, i+1)
    # train_random(i,i+1)
    # train_greedy(i,i+1)
    # print('\n')
# p =0.15 VM0-3 collecting data
# p =0.15 VM3-6 training
# p = 0.02 VM6-12 training
#
fetch_data(9,12)
statistics(M,K,t)






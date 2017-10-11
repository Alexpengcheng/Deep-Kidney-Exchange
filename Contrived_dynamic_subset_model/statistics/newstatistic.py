from csv import DictWriter, DictReader
import csv
import numpy as np
import matplotlib.pyplot as plt


class RandomGreedy():
    def __init__(self,data,policy):
        self.Vnc = 0
        self.Vco = 0
        self.data = data
        self.rewards = 0
        self.total_rewards = []
        self.trajectories = 0
        self.mean = 0
        self.std = 0
        self.policy = policy

    # Get the reward for this time tic
    def match(self,action):
        r = 0
        if action == 0: # no match
            r = 0
        elif action == 1: # match
            if self.Vco == 0:
                r = 0
            elif self.Vco >= self.Vnc:
                r = r + self.Vnc
                self.Vco = self.Vco - self.Vnc
                self.Vnc = 0
                r = r + self.Vco // 2
                self.Vco = self.Vco % 2

            else:
                r = self.Vco
                self.Vnc = self.Vnc - self.Vco
                self.Vco = 0

        return r

    def process(self):
        '''
        Process the whole file and collection the rewards for each trajectory and
        calculate the mean and std for these trajectories
        '''
        self.rewards = 0
        for line in self.data:
            if int(line['coming_Vnc']) ==0 and int(line['coming_Vco']) ==0:
                # clear the graph
                self.Vnc = 0
                self.Vco = 0
                self.total_rewards.append(self.rewards)
                self.rewards = 0
            else:
                self.Vnc += int(line['coming_Vnc'])
                self.Vco += int(line['coming_Vco'])

                if self.policy == 'random':
                    action = np.random.randint(0,2)
                elif self.policy == 'greedy':
                    action = 1
                else:
                    raise NotImplementedError

                self.rewards += self.match(action)

        # process the data
        data = np.asarray(self.total_rewards)
        self.mean = np.mean(data)
        self.std = np.std(data)

        return self.mean, self.std

class RL():
    def __init__(self,data):
        self.data = data
        self.rewards = 0
        self.total_rewards = []
        self.trajectories = 0
        self.mean = 0
        self.std = 0

    def process(self):
        self.rewards = 0
        for line in self.data:
            if int(line['coming_Vnc']) ==0 and int(line['coming_Vco']) ==0:
                self.rewards += int(line['reward'])
                self.total_rewards.append(self.rewards)
                self.rewards = 0
            else:
                self.rewards += int(line['reward'])

        # process the data
        data = np.asarray(self.total_rewards)
        self.mean = np.mean(data)
        self.std = np.std(data)

        return self.mean, self.std

class Upperbound():
    def __init__(self,data):
        self.Vnc = 0
        self.Vco = 0
        self.data = data
        self.rewards = 0
        self.total_rewards = []
        self.trajectories = 0
        self.mean = 0
        self.std = 0

    def match(self):
        if self.Vnc >= self.Vco:
            reward = self.Vco
        else:
            reward = self.Vnc + (self.Vco-self.Vnc)//2

        self.Vco = 0
        self.Vnc = 0

        return reward

    def process(self):
        self.rewards = 0
        for line in self.data:
            self.Vnc += int(line['coming_Vnc'])
            self.Vco += int(line['coming_Vco'])
            # print ('vco',self.Vco,'vnc',self.Vnc)

            if int(line['coming_Vnc']) == 0 and int(line['coming_Vco']) == 0:
                self.rewards = self.match()
                self.total_rewards.append(self.rewards)

        # process data
        data = np.asarray(self.total_rewards)
        self.mean = np.mean(data)
        self.std = np.std(data)

        return self.mean, self.std

class Optimal():
    def __init__(self,data,maxlength):
        self.Vnc = 0
        self.Vco = 0
        self.data = data
        self.length =maxlength
        self.rewards = np.zeros(self.length) # each reward is 200 time tics reward for this trajectory
        self.total_rewards = []
        self.trajectories = 0
        self.mean = 0
        self.std = 0
        self.tic = 0

    def match(self):
        '''
        match() here doesn't change Vco and Vnc
        :return:
        '''
        if self.Vnc >= self.Vco:
            reward = self.Vco
        else:
            reward = self.Vnc + (self.Vco-self.Vnc)//2

        return reward

    def process(self):
        self.rewards = np.zeros(self.length)
        for line in self.data:
            self.Vnc += int(line['coming_Vnc'])
            self.Vco += int(line['coming_Vco'])
            self.tic = int(line['time_tic'])

            # python array start from 0
            self.rewards[self.tic-1] = self.match()
            if int(line['coming_Vnc']) == 0 and int(line['coming_Vco']) == 0:
                self.rewards[self.tic-1:] = 0
                self.total_rewards.append(self.rewards)
                self.Vco = 0
                self.Vnc = 0
                self.rewards = np.zeros(self.length)

        batch = 0
        for reward in self.total_rewards:
            batch += reward

        bestpolicy = np.argmax(batch)

        # generate the best policy
        policyreward = []
        for trajec in self.total_rewards:
            policyreward.append(trajec[bestpolicy])

        # process the data
        data = np.asarray(policyreward)
        self.mean = np.mean(data)
        self.std = np.std(data)

        return self.mean, self.std





def summary():
    files = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]
    plist = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]


    for p in plist:
        up_mean = []
        rl_mean = []
        greedy_mean = []
        random_mean = []
        opt_mean = []

        up_std = []
        rl_std = []
        greedy_std = []
        random_std = []
        opt_std = []

        for file in files:
            rl_data = list(DictReader(open('../RL/p'+str(p)+'theta'+str(file)+'mean300_data.csv', 'r')))
            # greedy_data = list(DictReader(open('../greedy/p'+str(p)+'theta'+str(file)+'mean300_data.csv', 'r')))
            # random_data = list(DictReader(open('../random/p'+str(p)+'theta'+str(file)+'mean300_data.csv', 'r')))

            rlprocess = RL(rl_data)
            greedyprocess = RandomGreedy(rl_data,'greedy')
            randomprocess = RandomGreedy(rl_data,'random')
            upperprocess = Upperbound(rl_data)
            optprocess = Optimal(rl_data,350)

            # process the current file and add the mean and std
            rlmean, rlstd = rlprocess.process()
            greedymean, greedystd = greedyprocess.process()
            randmean, randstd = randomprocess.process()
            uppermean, upperstd = upperprocess.process()
            optmean, optstd = optprocess.process()

            rl_mean.append(rlmean)
            rl_std.append(rlstd)
            greedy_mean.append(greedymean)
            greedy_std.append(greedystd)
            random_mean.append(randmean)
            random_std.append(randstd)
            up_mean.append(uppermean)
            up_std.append(upperstd)
            opt_mean.append(optmean)
            opt_std.append(optstd)


        # print ('opt:',up_mean)
        print ('rl:',rl_mean)
        print ('opt:',opt_mean)
        # print ('greedy', (greedy_mean))
        # print ('random', (random_mean))
        print ('files',len(files))
        fig = plt.figure()
        rl = plt.errorbar (files, np.asarray(rl_mean), yerr=np.asarray(rl_std), label = 'RL_TRPO', fmt='-o',capsize=4)
        up= plt.errorbar(files, up_mean, yerr=up_std, label='Cheat',fmt='-o',capsize=4)
        greedy = plt.errorbar(files, greedy_mean, yerr=greedy_std, label='Greedy',fmt='-o',capsize=4)
        random = plt.errorbar(files, random_mean, yerr=random_std, label='Random',fmt='-o',capsize=4)
        opt = plt.errorbar(files, opt_mean, yerr=opt_std, label='Single_Match', fmt='-o', capsize=4)


        plt.legend(handles=[rl, up, greedy, random, opt])
        plt.title("p = "+str(p))
        # # without rl
        # plt.legend(handles=[ opt, greedy, random])

        plt.xticks(np.arange(0, 1, 0.1))
        plt.ylabel('Average Rewards')
        plt.xlabel('Theta')
        fig.savefig("p"+str(p)+'theta'+'mean300_data.png')

summary()
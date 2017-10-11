from csv import DictWriter, DictReader
import csv
import numpy as np
import matplotlib.pyplot as plt





class Stat:
    def __init__(self,data):
        self.Vnc = 0
        self.Vco = 0
        self.data = data
        self.rewards = 0
        self.trajectories = 0
        self.result = 0

    def match(self):
        if self.Vnc >= self.Vco:
            reward = self.Vco
        else:
            reward = self.Vnc + (self.Vco-self.Vnc)//2

        self.Vco = 0
        self.Vnc = 0

        return reward

    def process(self):
        trajectories = 0
        rewards = 0
        for line in self.data:
            self.result = float(line['ave_rewards'])
            self.trajectories = int(line['trajectory'])
            if int(line['coming_Vnc']) != 0:
                self.Vnc += int(line['coming_Vnc'])
                self.Vco += int(line['coming_Vco'])
                # print ('vco',self.Vco,'vnc',self.Vnc)

            else:
                reward = self.match()
                rewards += reward
                # print (reward)

        ave_reward = rewards/self.trajectories

        return ave_reward

    def strategy_result(self):

        return self.result

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
            else:
                self.rewards += int(line['reward'])

        # process the data
        data = np.asarray(self.total_rewards)
        self.mean = np.mean(data)
        self.std = np.std(data)

        return self.mean, self.std

class Upperbound():
    def __init__(self,data):
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




def summary():
    files = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]
    plist = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]


    for p in plist:
        opt = []
        rl = []
        greedy = []
        random = []
        for file in files:
            rl_data = list(DictReader(open('../RL/p'+str(p)+'theta'+str(file)+'mean300_data.csv', 'r')))
            greedy_data = list(DictReader(open('../greedy/p'+str(p)+'theta'+str(file)+'mean300_data.csv', 'r')))
            random_data = list(DictReader(open('../random/p'+str(p)+'theta'+str(file)+'mean300_data.csv', 'r')))

            pro = Stat(rl_data)
            pro_greedy = Stat(greedy_data)
            pro_random = Stat(random_data)

            opt.append(pro.process())
            rl.append(pro.strategy_result())

            pro_greedy.process()
            greedy.append(pro_greedy.strategy_result())

            pro_random.process()
            random.append(pro_random.strategy_result())

        print ('opt:',opt)
        print ('rl:',rl)
        print ('greedy', greedy)
        print ('random', random)
        fig = plt.figure()

        rl, = plt.plot (files, rl, label = 'RL_TRPO')
        opt, = plt.plot(files, opt, label='Optimal')
        greedy, = plt.plot(files, greedy, label='Greedy')
        random, = plt.plot(files, random, label='Random')

        plt.legend(handles=[rl, opt, greedy, random])

        # # without rl
        # plt.legend(handles=[ opt, greedy, random])

        plt.xticks(np.arange(0, 1, 0.1))
        plt.ylabel('Average Rewards')
        plt.xlabel('Theta')
        fig.savefig("p"+str(p)+'theta'+'mean300_data.png')

summary()
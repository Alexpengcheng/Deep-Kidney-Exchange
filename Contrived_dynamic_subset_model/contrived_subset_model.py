import numpy as np
from scipy import stats as st

class Contrived_subset():
    def __init__(self, N, p, expo):
        # default parameters:
        self.__N = N # incoming vertices each time point
        self.__p = p # Vcon distribution
        self.__expo = expo # reset distribution, large value indicate reset often
        self.__t = 1 # time point
        self.__Vco = 0.0 # compatible vertices
        self.__Vnc = 0.0 # non_compatible vertices
        self.__reward = 0.0 # rewards
        self.reset_time = 100
        self.__done = False

    def reset(self):
        '''
        clear the present graph and restart a new time period
        '''
        self.__Vco = 0
        self.__Vnc = 0
        self.perent_co = st.geom.pmf(k=1, p=self.__p, loc=0)
        add_co = int(self.__N * self.perent_co)
        add_nc = self.__N - add_co
        self.__Vnc = self.__Vnc + add_nc
        self.__Vco = self.__Vco + add_co
        self.__t = 1
        self.__reward = 0
        self.__done = False
        self.reset_time = 1 + int(np.random.geometric(self.__expo))
        return np.array([self.__Vco,self.__Vnc])/3000

    def match(self):
        '''
        if action==1, maximum match for the present graph
        else action==0, don't match anything
        notes: non_compatible vertices will be matched with priority
        '''
        self.__reward = 0

        # no compatible vertices, match nothing
        if self.__Vco == 0:
            self.__reward = 0

        elif self.__Vco >= self.__Vnc:
            self.__reward = self.__reward + self.__Vnc
            self.__Vco = self.__Vco - self.__Vnc
            self.__Vnc = 0
            self.__reward = self.__reward + self.__Vco //2
            self.__Vco = self.__Vco %2

        else:
            self.__reward = self.__Vco
            self.__Vco = 0
            self.__Vnc = self.__Vnc - self.__Vco

        return self.__reward

    def step(self,action):
        '''
        sequential actions:
        1. introduce the incoming vertices
        2. match and return rewards
        3. increase time t
        '''

        # randp = np.random.rand()
        # expop = st.expon.cdf(x=self.__t, loc=0, scale=1.0/self.__expo)
        # if randp < expop:
        #     self.reset()

        if self.__t < self.reset_time:

            if action == 1:
                rewards = self.match()
            elif action == 0:
                rewards = 0
            else:
                print("illegal action!")
                raise NotImplementedError

            # generating new observation
            self.__t = self.__t + 1
            if self.__t == self.reset_time:
                self.done = True
                # self.__Vco = 0
                # self.__Vnc = 0
            else:
                self.perent_co = st.geom.pmf(k=self.__t, p=self.__p, loc=0)
                add_co = int(self.__N * self.perent_co)
                add_nc = self.__N - add_co
                self.__Vnc = self.__Vnc + add_nc
                self.__Vco = self.__Vco + add_co
                self.done = False
        else:
            rewards = 0
            self.__done = True
        observation = np.array([self.__Vco,self.__Vnc])/(self.__N*0.5/self.__expo)
        return observation, rewards, self.__done, "none"


def main():
    envs = Contrived_subset(N=300, p=0.7, expo=0.05)
    total_rewards = []
    Vco_percent = []
    for i in range (5000):
        done=False
        oba=envs.reset()
        print ("init",oba)
        rewards=0
        j=0
        while done==False:
            # action=np.random.randint(0,2)
            # action = 1
            if j==0:
                action=0
                j=1
            else:
                action=1
            obsr,re,done,_ = envs.step(action)
            # print ("iter",i,"obsr",obsr)
            rewards = rewards + re
        print ("rewards", rewards)
        total_rewards.append(rewards)
        # print ("iter",i,"rewards",rewards)
    mean = np.sum(total_rewards)/len(total_rewards)
    print ("mean",mean)
    print (len(total_rewards))



if __name__ == '__main__':
    main()
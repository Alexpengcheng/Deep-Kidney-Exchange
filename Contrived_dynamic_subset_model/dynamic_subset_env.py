import logging
import numpy as np
from scipy import stats as st
from gym import spaces
import gym
from gym.utils import seeding

logger = logging.getLogger(__name__)

class Contrived_subset(gym.Env):
    def __init__(self):
        # default parameters:
        self.__N = 300 # incoming vertices each time point
        self.__p = 0.7 # Vcon distribution
        self.__expo = 0.05 # reset distribution, large value indicate reset often
        self.__t = 1 # time point
        self.__Vco = 0.0 # compatible vertices
        self.__Vnc = 0.0 # non_compatible vertices
        self.__reward = 0.0 # rewards
        self.reset_time = 100
        self.__done = False

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            (2,))

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
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

    def _match(self):
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

        return self.__reward/self.reset_time

    def _step(self,action):
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
                rewards = self._match()
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

        return np.array([self.__Vco,self.__Vnc])/3000, rewards, self.__done, {}

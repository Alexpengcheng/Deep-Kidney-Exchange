import logging
import numpy as np
from scipy import stats as st
from gym import spaces
import gym
from gym.utils import seeding
from csv import DictWriter

logger = logging.getLogger(__name__)

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
        self.N = 300 # incoming vertices each time point
        self.mean = 300
        self.std = 20
        self.coming_total = self.N
        self.p = 0.7 # Vcon distribution
        self.expo = 0.05 # reset distribution, large value indicate reset often
        self.t = 1 # time tiac
        self.trajectory = 0 # trajectory number
        self.Vco = 0.0 # compatible vertices
        self.Vnc = 0.0 # non_compatible vertices
        self.reward = 0.0 # rewards
        self.reset_time = 100
        self.done = False
        self.record = False # indicate if record the environment details
        self.path = './p%stheta%smean%s_data.csv' %(self.p,self.expo,self.N)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            (2,))

        self.ave_rewards = 0

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        '''
        clear the present graph and restart a new time period
        '''
        self.Vco = 0
        self.Vnc = 0
        self.perent_co = st.geom.pmf(k=1, p=self.p, loc=0)
        self.N = int(np.random.normal(self.mean, self.std))
        if self.N <= 0:
            self.N = 0
        add_co = int(self.N * self.perent_co)
        add_nc = self.N - add_co
        self.Vnc = self.Vnc + add_nc
        self.Vco = self.Vco + add_co
        self.t = 1
        self.reward = 0
        self.done = False
        self.reset_time = 1 + int(np.random.geometric(self.expo))

        # increase trajectory number
        self.trajectory += 1

        d = {'trajectory': self.trajectory,
             "time_tic": self.t,
             "currt_Vco": self.Vco,
             "currt_Vnc": self.Vnc,
             "coming_total_V": self.N,
             "coming_Vco": add_co,
             "coming_Vnc": add_nc,
             "reward": 0,
             "mean": self.mean,
             "std": self.std,
             "p": self.p,
             "theta": self.expo,
             "ave_rewards": self.ave_rewards/self.trajectory
             }
        if self.record == True:
            self._record(d)


        return np.array([self.Vco,self.Vnc])/3000

    def _match(self):
        '''
        if action==1, maximum match for the present graph
        else action==0, don't match anything
        notes: non_compatible vertices will be matched with priority
        '''
        self.reward = 0

        # no compatible vertices, match nothing
        if self.Vco == 0:
            self.reward = 0

        elif self.Vco >= self.Vnc:
            self.reward = self.reward + self.Vnc
            self.Vco = self.Vco - self.Vnc
            self.Vnc = 0
            self.reward = self.reward + self.Vco //2
            self.Vco = self.Vco %2

        else:
            self.reward = self.Vco
            self.Vnc = self.Vnc - self.Vco
            self.Vco = 0

        return self.reward

    def _step(self,action):
        '''
        sequential actions:
        1. match and return rewards
        2. introduce the incoming vertices
        3. increase time t
        '''

        # randp = np.random.rand()
        # expop = st.expon.cdf(x=self.__t, loc=0, scale=1.0/self.__expo)
        # if randp < expop:
        #     self.reset()

        if self.t < self.reset_time:

            if action == 1:
                rewards = self._match()
            elif action == 0:
                rewards = 0
            else:
                print("illegal action!")
                raise NotImplementedError

            # generating new observation
            self.t = self.t + 1
            if self.t == self.reset_time:
                self.done = True
                add_nc = 0
                add_co = 0
                # self.__Vco = 0
                # self.__Vnc = 0
            else:
                self.perent_co = st.geom.pmf(k=self.t, p=self.p, loc=0)
                self.N = int(np.random.normal(self.mean,self.std))
                if self.N <= 0:
                    self.N = 0
                add_co = int(self.N * self.perent_co)
                add_nc = self.N - add_co
                self.Vnc = self.Vnc + add_nc
                self.Vco = self.Vco + add_co
                self.done = False
        else:
            rewards = 0
            add_co = 0
            add_nc = 0
            self.done = True

        self.ave_rewards =(self.ave_rewards+rewards)

        d = {'trajectory': self.trajectory,
             "time_tic": self.t,
             "currt_Vco": self.Vco,
             "currt_Vnc": self.Vnc,
             "coming_total_V": self.N,
             "coming_Vco": add_co,
             "coming_Vnc":add_nc,
             "reward": rewards,
            "mean": self.mean,
             "std": self.std,
            "p": self.p,
            "theta": self.expo,
             "ave_rewards": self.ave_rewards/self.trajectory
             }
        if self.record == True:
            self._record(d)

        return np.array([self.Vco,self.Vnc])/3000, rewards, self.done, {}

    def _record(self,d):
        o = DictWriter(open(self.path, 'a'), ["trajectory", "time_tic","currt_Vco","currt_Vnc",
                                              "coming_total_V","coming_Vco","coming_Vnc","reward",
                                              "mean","std","p","theta","ave_rewards"])
        if self.trajectory ==1 and self.t== 1 :
            o.writeheader()

        o.writerow(d)



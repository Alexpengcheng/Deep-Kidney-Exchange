import gym
from gym import Wrapper
from csv import DictWriter

class SubsetWrapper(Wrapper):
    def __init__(self,env,para):
        env = env.unwrapped

        # model parameters
        if 'total_vertices' in para: env.N = para.pop('total_vertices')
        if 'p' in para: env.p = para.pop("p")
        if 'theta' in para: env.expo = para.pop("theta")
        if 'mean' in para: env.mean = para.pop("mean")
        if 'std' in para: env.std = para.pop("std")
        if 'record' in para: env.record = para.pop("record")
        if 'path' in para: env.path = para.pop("path")
        env.trajectory = 0

        super(SubsetWrapper, self).__init__(env)
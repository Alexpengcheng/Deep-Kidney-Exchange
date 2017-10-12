#!/usr/bin/python
import os

'''
Automatically distribute the RL experiment
'''

M= [32, 64, 128, 256]
K= [32, 128, 512]
CYCLE_CAP= [2, 3, 4]
CHAIN_CAP= [0, 3, 6, 9]
SAMPLE = 5
CHAINSAMPLE = 3
CYCLESAMPLE = 3
TAU = 5
ALPHA = 0.05


for m in M:
    for k in K:
        for cycle in CYCLE_CAP:
            for chain in CHAIN_CAP:
                os.system('nohup python3 kidex_trpo.py '
                        '--cyclecap %s '
                        '--chaincap %s '
                        '--M %s '
                        '--K %s '
                        '--sample %s '
                        '--chainsample %s '
                        '--cyclesampl %s '
                        '--tau %s '
                        '--alpha %s '
                          '>M%sK%sChain%sCycle%s &' % (cycle, chain, m, k, SAMPLE, CHAINSAMPLE, CYCLESAMPLE, TAU, ALPHA,
                                                       m,k,chain,cycle))
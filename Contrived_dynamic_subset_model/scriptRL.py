#!/usr/bin/python
import os

plist=[0.1, 0.2]
thetas = [ 0.02, 0.05, 0.1, 0.15, 0.2, 0.25 ,0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]

for p in plist:
    for theta in thetas:
        if theta <= 0.25:
            os.system('nohup python3 contrived_subset_trpo.py --p %s --theta %s >p%stheta%s &' % (p,theta,p,theta))
        elif theta < 0.6:
            os.system('nohup python3 contrived_subset_trpo.py --p %s --theta %s --batchsize 2000 --trainsize 40000 >p%stheta%s &' % (p,theta,p,theta))
        elif theta <= 0.9:
            os.system('nohup python3 contrived_subset_trpo.py --p %s --theta %s --batchsize 1500 --trainsize 55000 >p%stheta%s &' % (p, theta,p,theta))



# After finish training

for p in plist:
    for theta in thetas:
        os.system('nohup python3 contrived_subset_trpo.py --p %s --theta %s --trainsize 1000 --loadmodel True --record True >sta &' % (p, theta))


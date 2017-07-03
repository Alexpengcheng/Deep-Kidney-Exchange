from AsynAC_kd import*
import numpy as np
import tensorflow as tf

'''Notes:
The Maindiriver.py works as the top level control of the Kidney Exchange process so that we don't need to
call every module in shell, which is too expensive for computing. You need to comment out the last line code
"main_nnet()" in AsynAC_kd.py. The following is a short example, I manually write the reward into the reward file:
'''

#Training Constants
MAXITERATION=20 #Determine the iterations we want to train te network

for i in range(MAXITERATION):
        main_nnet()
        with open('obj_value.txt', 'a') as g:
                print [int(i)]
                np.savetxt(g,[int(i)])

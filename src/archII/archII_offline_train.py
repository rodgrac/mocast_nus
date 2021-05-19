import os
import sys
import time

os.system("nohup sh -c '" +
          sys.executable + " archII_train.py > /scratch/rodney/models/nuScenes/logs/may3_stse_v2_ortho.txt" +
                           "' &")
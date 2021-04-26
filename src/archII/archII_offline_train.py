import os
import sys
import time

os.system("nohup sh -c '" +
          sys.executable + " archII_train.py > /scratch/rodney/models/nuScenes/logs/apr25_stse_ortho_h2s_p3s.txt" +
                           "' &")
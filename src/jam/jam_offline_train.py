import os
import sys
import time

os.system("nohup sh -c '" +
          sys.executable + " train_jam.py > /scratch/rodney/models/nuScenes/logs/tfrjam_mar21_tfr.txt" +
                           "' &")
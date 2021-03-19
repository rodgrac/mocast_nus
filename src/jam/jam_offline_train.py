import os
import sys
import time

os.system("nohup sh -c '" +
          sys.executable + " train_jam.py > /scratch/rodney/models/nuScenes/logs/tfrjam_mar19_att2_fftdec.txt" +
                           "' &")
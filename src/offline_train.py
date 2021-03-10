import os
import sys

os.system("nohup sh -c '" +
          sys.executable + " train.py > /scratch/rodney/models/nuScenes/log_mocast4_ortho.txt" +
                           "' &")

import os
import sys

os.system("nohup sh -c '" +
          sys.executable + " train.py > /scratch/rodney/models/nuScenes/logs/mocast4_mar17_clsenc_cat.txt" +
                           "' &")

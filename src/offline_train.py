import os
import sys

os.system("nohup sh -c '" +
          sys.executable + " train.py > /scratch/rodney/models/nuScenes/logs/mocast4_apr12_basicdec.txt" +
                           "' &")

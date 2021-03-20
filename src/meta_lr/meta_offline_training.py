import os
import sys
import time

os.system("nohup sh -c '" +
          sys.executable + " meta_train.py --batch_size 16 --gpu 2 > /scratch/rodney/models/nuScenes/logs/metalr_mar19_pretrain_fut.txt" +
                           "' &")
# time.sleep(5)
#
# os.system("nohup sh -c '" +
#           sys.executable + " meta_train.py --batch_size 8 --gpu 1 > /scratch/rodney/models/nuScenes/log1.txt" +
#                            "' &")

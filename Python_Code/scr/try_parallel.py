
import math
import torch
import gpytorch
import sys
from matplotlib import pyplot as plt
sys.path.append('../')


n_devices = torch.cuda.device_count()

print(torch.cuda.is_available())
print('Planning to run on {} GPUs.'.format(n_devices))

import multiprocessing as mp
print("Number of processors: ", mp.cpu_count())
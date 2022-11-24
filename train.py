import os
import cv2
import random
import numpy as np
import torch
from src.misf import MISF_train




def main():


    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.backends.cudnn.benchmark = True   # cudnn auto-tuner

    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0)

    # initialize random seed
    torch.manual_seed(10)
    torch.cuda.manual_seed_all(10)
    np.random.seed(10)
    random.seed(10)


    # build the model and initialize
    max_epoch = 2
    model = MISF_train(max_epoch)
    # model = MISF_train()
    print('\nstart training...\n')
    model.train()



if __name__ == "__main__":
    main()
import os
import cv2
import random
import numpy as np
import torch
from src.misf import MISF_test




def main():



    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # config.DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True   # cudnn auto-tuner

    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0)

    # initialize random seed
    torch.manual_seed(10)
    torch.cuda.manual_seed_all(10)
    np.random.seed(10)
    random.seed(10)


    # build the model and initialize
    model = MISF_test()


    # model testing
    print('\nstart testing...\n')
    gen_weights_path = './checkpoints/celebA_InpaintingModel_gen.pth'
    dis_weights_path = './checkpoints/celebA_InpaintingModel_dis.pth'
    # gen_weights_path = './checkpoints/50_InpaintingModel_gen.pth'
    # dis_weights_path = './checkpoints/50_InpaintingModel_dis.pth'
    model.load(gen_weights_path,dis_weights_path)
    model.test()





if __name__ == "__main__":
    main()
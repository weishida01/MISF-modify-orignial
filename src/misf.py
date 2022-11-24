import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import Dataset
from .models import InpaintingModel
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import kpn.utils as kpn_utils
from .utils import Progbar
import time


class MISF_train():
    def __init__(self,max_epoch=500):
        self.max_epoch = max_epoch
        self.model_name = 'inpaint'
        self.inpaint_model = InpaintingModel().to(torch.device("cuda"))

        TRAIN_FLIST = "./data/train_flist.txt"
        VAL_FLIST = "./data/valid_flist.txt"
        TRAIN_MASK_FLIST = "./data/mask-train_flist.txt"
        VAL_MASK_FLIST = "./data/mask-valid_flist.txt"

        self.train_dataset = Dataset( TRAIN_FLIST, TRAIN_MASK_FLIST, training=True)
        self.val_dataset = Dataset( VAL_FLIST, VAL_MASK_FLIST, training=False)
        print('—'*20)
        print('max_epoch:',self.max_epoch)
        print('train dataset:{}'.format(len(self.train_dataset)))
        print('eval dataset:{}'.format(len(self.val_dataset)))
        print('—' * 20)

        inpaint_path = os.path.join('./checkpoints', 'inpaint')
        self.results_path = os.path.join(inpaint_path, 'results')
        self.log_file = os.path.join(inpaint_path, time.strftime('%Y-%m-%d-%H-%M')+'_inpaint.log')

        if not os.path.exists('./checkpoints'):
            os.mkdir('./checkpoints')
        if not os.path.exists(inpaint_path):
            os.mkdir(inpaint_path)
        if not os.path.exists(self.results_path):
            os.mkdir(self.results_path)

    def save(self,epoch):
        self.inpaint_model.save(epoch)

    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=4,
            num_workers=0,
            drop_last=True,
            shuffle=True
        )

        epoch = 0
        max_psnr = 0
        total = len(self.train_dataset)
        while(epoch < self.max_epoch):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])

            for items in train_loader:

                self.inpaint_model.train()

                images, masks = self.cuda(*items)

                # inpaint model
                # train
                outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, masks)
                outputs_merged = (outputs * masks) + images * (1 - masks)

                # backward
                self.inpaint_model.backward(gen_loss, dis_loss)
                iteration = self.inpaint_model.iteration

                logs = [("epoch", epoch),("iter", iteration),] + logs
                self.log(logs)

                progbar.add(len(images),values=logs)


            # save model at checkpoints
            # evaluate model at checkpoints
            if epoch % 2 == 0:
                self.save(epoch)

                print('\nstart eval...\n')
                cur_psnr = self.eval(epoch)
                if cur_psnr > max_psnr:
                    max_psnr = cur_psnr

            self.inpaint_model.iteration = 0

        print('\nEnd training....')

    def eval(self,epoch):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=1,
            drop_last=True,
            shuffle=False
        )

        self.inpaint_model.eval()

        psnr_all = []
        ssim_all = []
        l1_list = []

        iteration = self.inpaint_model.iteration
        with torch.no_grad():
            for items in val_loader:
                images,masks = self.cuda(*items)

                # inpaint model
                # eval
                outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, masks)
                outputs_merged = (outputs * masks) + images * (1 - masks)

                results_path = os.path.join(self.results_path, 'val')
                if self.inpaint_model.iteration % 3 == 0:
                    img_list2 = [images * (1 - masks), outputs_merged, outputs, images]
                    name_list2 = ['in', 'pred_2', 'pre_1', 'gt']
                    kpn_utils.save_sample_png(sample_folder=results_path,
                                              sample_name='epoch_{}_{}'.format(epoch, 0),
                                              img_list=img_list2,
                                              name_list=name_list2, pixel_max_cnt=255, height=-1,
                                              width=-1)

                psnr, ssim = self.metric(images, outputs_merged)
                psnr_all.append(psnr)
                ssim_all.append(ssim)

                l1_loss = torch.nn.functional.l1_loss(outputs_merged, images, reduction='mean').item()
                l1_list.append(l1_loss)
                print('psnr:{}/{}  ssim:{}/{} l1:{}/{}  {}/{}'.format(psnr, np.average(psnr_all),
                                                                      ssim, np.average(ssim_all),
                                                                      l1_loss, np.average(l1_list),
                                                                      len(psnr_all), len(self.val_dataset)))

            print('iteration:{} ave_psnr:{}  ave_ssim:{} ave_l1:{}'.format(
                iteration,
                np.average(psnr_all),
                np.average(ssim_all),
                np.average(l1_list),
            ))

            return np.average(psnr_all)

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(torch.device("cuda")) for item in args)


    def metric(self, gt, pre):
        pre = pre.clamp_(0, 1) * 255.0
        pre = pre.permute(0, 2, 3, 1)
        pre = pre.detach().cpu().numpy().astype(np.uint8)[0]

        gt = gt.clamp_(0, 1) * 255.0
        gt = gt.permute(0, 2, 3, 1)
        gt = gt.cpu().detach().numpy().astype(np.uint8)[0]

        psnr = min(100, compare_psnr(gt, pre))
        ssim = compare_ssim(gt, pre, multichannel=True, data_range=255)

        return psnr, ssim


class MISF_test():
    def __init__(self):
        self.model_name = 'inpaint'
        self.inpaint_model = InpaintingModel().to(torch.device("cuda"))

        Test_FLIST = "./data/test_flist.txt"
        Test_MASK_FLIST = "./data/mask-test_flist.txt"
        self.test_dataset = Dataset(Test_FLIST, Test_MASK_FLIST, training=True)
        print('test dataset:{}'.format(len(self.test_dataset)))

        inpaint_path = os.path.join('./checkpoints', 'inpaint')
        self.samples_path = os.path.join(inpaint_path, 'samples')
        self.results_path = os.path.join(inpaint_path, 'results')
        self.log_file = os.path.join(inpaint_path, 'log_' + 'inpaint' + '.log')

        if not os.path.exists('./checkpoints'):
            os.mkdir('./checkpoints')
        if not os.path.exists(inpaint_path):
            os.mkdir(inpaint_path)
        if not os.path.exists(self.samples_path):
            os.mkdir(self.samples_path)
        if not os.path.exists(self.results_path):
            os.mkdir(self.results_path)

    def load(self,gen_weights_path,dis_weights_path):
        self.inpaint_model.load(gen_weights_path,dis_weights_path)


    def test(self):
        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            drop_last=True,
            shuffle=False
        )

        self.inpaint_model.eval()

        psnr_all = []
        ssim_all = []
        l1_list = []

        iteration = self.inpaint_model.iteration
        with torch.no_grad():
            for items in test_loader:
                images, masks = self.cuda(*items)

                # inpaint model
                # eval
                outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, masks)
                outputs_merged = (outputs * masks) + images * (1 - masks)

                psnr, ssim = self.metric(images, outputs_merged)
                psnr_all.append(psnr)
                ssim_all.append(ssim)

                l1_loss = torch.nn.functional.l1_loss(outputs_merged, images, reduction='mean').item()
                l1_list.append(l1_loss)

                # sample
                results_path = os.path.join(self.results_path, 'test')
                img_list2 = [images * (1 - masks), outputs_merged, outputs, images]
                name_list2 = ['in', 'pred2', 'pre1', 'gt']
                kpn_utils.save_sample_png(sample_folder=results_path,
                                          sample_name='ite_{}_{}'.format(iteration, len(psnr_all)),
                                          img_list=img_list2,
                                          name_list=name_list2, pixel_max_cnt=255, height=-1,
                                          width=-1)

                print('psnr:{}/{}  ssim:{}/{} l1:{}/{}  {}/{}'.format(psnr, np.average(psnr_all),
                                                                      ssim, np.average(ssim_all),
                                                                      l1_loss, np.average(l1_list),
                                                                      len(psnr_all), len(self.test_dataset)))

            print('iteration:{} ave_psnr:{}  ave_ssim:{} ave_l1:{}'.format(
                iteration,
                np.average(psnr_all),
                np.average(ssim_all),
                np.average(l1_list),
            ))

            return np.average(psnr_all)

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(torch.device("cuda")) for item in args)

    def metric(self, gt, pre):
        pre = pre.clamp_(0, 1) * 255.0
        pre = pre.permute(0, 2, 3, 1)
        pre = pre.detach().cpu().numpy().astype(np.uint8)[0]

        gt = gt.clamp_(0, 1) * 255.0
        gt = gt.permute(0, 2, 3, 1)
        gt = gt.cpu().detach().numpy().astype(np.uint8)[0]

        psnr = min(100, compare_psnr(gt, pre))
        ssim = compare_ssim(gt, pre, multichannel=True, data_range=255)

        return psnr, ssim
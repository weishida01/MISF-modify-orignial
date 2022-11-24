import os
import torch
import torch.nn as nn
import torch.optim as optim
from .networks import InpaintGenerator, Discriminator
from .loss import AdversarialLoss, PerceptualLoss, StyleLoss

class BaseModel(nn.Module):
    def __init__(self, name):
        super(BaseModel, self).__init__()
        self.name = name
        self.iteration = 0
        if not os.path.exists('./checkpoints'):
            os.mkdir('./checkpoints')
        if not os.path.exists('./checkpoints/inpaint'):
            os.mkdir('./checkpoints/inpaint')
        if not os.path.exists('./checkpoints/inpaint/pth'):
            os.mkdir('./checkpoints/inpaint/pth')
        self.model_save = './checkpoints/inpaint/pth'


    def load(self, gen_weights_path,dis_weights_path):
        self.gen_weights_path = gen_weights_path
        self.dis_weights_path = dis_weights_path

        data = torch.load(self.gen_weights_path)
        self.generator.load_state_dict(data['generator'])
        self.iteration = data['iteration']

        data = torch.load(self.dis_weights_path)
        self.discriminator.load_state_dict(data['discriminator'])

    def save(self,epoch):
        generate_param = self.generator.state_dict()
        dis_param = self.discriminator.state_dict()
        print('save...single GPU')

        torch.save({
            'iteration': self.iteration,
            'generator': generate_param
        }, os.path.join(self.model_save, '{}_{}_gen.pth'.format(epoch, self.name)))

        torch.save({
            'discriminator': dis_param
        }, os.path.join(self.model_save, '{}_{}_dis.pth'.format(epoch, self.name)))

        print('\nsaving %s...\n' % self.name)


class InpaintingModel(BaseModel):
    def __init__(self):
        super(InpaintingModel, self).__init__('InpaintingModel')

        # generator input: [rgb(3) + edge(1)]
        # discriminator input: [rgb(3)]
        generator = InpaintGenerator()
        discriminator = Discriminator(in_channels=3, use_sigmoid=True)

        l1_loss = nn.L1Loss()
        perceptual_loss = PerceptualLoss()
        style_loss = StyleLoss()
        adversarial_loss = AdversarialLoss(type='nsgan')

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('perceptual_loss', perceptual_loss)
        self.add_module('style_loss', style_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        LR = 0.0001
        BETA1 = 0.0
        BETA2 = 0.9
        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(LR),
            betas=(BETA1, BETA2)
        )

        D2G_LR = 0.1
        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(LR) * float(D2G_LR),
            betas=(BETA1, BETA2)
        )

    def process(self, images, masks):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()


        # process outputs
        outputs = self(images, masks)
        gen_loss = 0
        dis_loss = 0


        # discriminator loss
        dis_input_real = images
        dis_input_fake = outputs.detach()
        dis_real, _ = self.discriminator(dis_input_real)                    # in: [rgb(3)]
        dis_fake, _ = self.discriminator(dis_input_fake)                    # in: [rgb(3)]
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2


        # generator adversarial loss
        INPAINT_ADV_LOSS_WEIGHT = 0.1
        gen_input_fake = outputs
        gen_fake, _ = self.discriminator(gen_input_fake)                    # in: [rgb(3)]
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss


        # generator l1 loss
        L1_LOSS_WEIGHT = 1
        gen_l1_loss = self.l1_loss(outputs, images) * L1_LOSS_WEIGHT / torch.mean(masks)
        gen_loss += gen_l1_loss


        # generator perceptual loss
        CONTENT_LOSS_WEIGHT = 0.1
        gen_content_loss = self.perceptual_loss(outputs, images)
        gen_content_loss = gen_content_loss * CONTENT_LOSS_WEIGHT
        gen_loss += gen_content_loss


        # generator style loss
        STYLE_LOSS_WEIGHT = 250
        gen_style_loss = self.style_loss(outputs * masks, images * masks)
        gen_style_loss = gen_style_loss * STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss


        # create logs
        logs = [
            ("l_d2", dis_loss.item()),
            ("l_g2", gen_gan_loss.item()),
            ("l_l1", gen_l1_loss.item()),
            ("l_per", gen_content_loss.item()),
            ("l_sty", gen_style_loss.item()),
        ]

        return outputs, gen_loss, dis_loss, logs

    def forward(self, images, masks):
        # images_masked = (images * (1 - masks).float()) + masks
        images_masked = images * (1 - masks)
        inputs = torch.cat((images_masked, masks), dim=1)
        outputs = self.generator(inputs)      # in: [rgb(3) + edge(1)]
        return outputs

    def backward(self, gen_loss=None, dis_loss=None):
        gen_loss.backward()
        self.gen_optimizer.step()

        dis_loss.backward()
        self.dis_optimizer.step()

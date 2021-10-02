import torch
import torch.nn as nn

class GANLoss:
    """ Base class for all losses
        @args:
            dis: Discriminator used for calculating the loss
                 Note this must be a part of the GAN framework
    """

    def __init__(self, dis):
        self.dis = dis

    def dis_loss(self, real_samps, fake_samps):
        """
        calculate the discriminator loss using the following data
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("dis_loss method has not been implemented")

    def gen_loss(self, real_samps, fake_samps):
        """
        calculate the generator loss
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("gen_loss method has not been implemented")

class StandardGAN(GANLoss):

    def __init__(self, dis, real_label=0.9, fake_label=0.0):
        super().__init__(dis)

        # define the criterion and activation used for object
        self.criterion = nn.BCEWithLogitsLoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.real_label = real_label
        self.fake_label = fake_label

    def dis_loss(self, real_samps, fake_samps):
        # predictions for real images and fake images separately :
        r_preds = self.dis(real_samps)
        f_preds = self.dis(fake_samps)

        # calculate the real loss:
        real_labels = torch.full((r_preds.shape[0],), self.real_label, dtype=torch.float, device=self.device)
        r_loss_preds = torch.squeeze(r_preds) if torch.numel(r_preds) > 1 else r_preds
        real_loss = self.criterion(r_loss_preds, real_labels)

        # calculate the fake loss:
        fake_labels = torch.full((f_preds.shape[0],), self.fake_label, dtype=torch.float, device=self.device)
        f_loss_preds = torch.squeeze(f_preds) if torch.numel(f_preds) > 1 else f_preds
        fake_loss = self.criterion(f_loss_preds, fake_labels)

        # return final losses
        return (real_loss + fake_loss) / 2

    def gen_loss(self, _, fake_samps):
        preds = self.dis(fake_samps)
        labels = torch.full((preds.shape[0],), self.real_label, dtype=torch.float, device=self.device)
        loss_preds = torch.squeeze(preds) if torch.numel(preds) > 1 else preds
        return self.criterion(loss_preds, labels)
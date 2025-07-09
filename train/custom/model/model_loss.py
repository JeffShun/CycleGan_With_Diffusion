
import torch
import torch.nn as nn
        
class GeneratorLoss(nn.Module):
    def __init__(self, consist_loss_weight=10.0, adv_loss_weight=1.0, id_loss_weight=2.0):
        super(GeneratorLoss, self).__init__()
        self.consist_loss_f = nn.L1Loss()
        self.adv_loss_f = nn.MSELoss()   
        self.id_loss_f = nn.L1Loss()
        self.consist_loss_weight = consist_loss_weight
        self.adv_loss_weight = adv_loss_weight
        self.id_loss_weight = id_loss_weight

    def forward(self, fake_img, src, idt, rec, discriminator, flag):
        fake_score = discriminator(fake_img)
        consist_loss = self.consist_loss_f(rec, src)
        adv_loss = self.adv_loss_f(fake_score, torch.ones_like(fake_score))
        id_loss = self.id_loss_f(idt, src)

        return {f"consist_loss_{flag}": self.consist_loss_weight * consist_loss, 
                f"adv_loss_{flag}": self.adv_loss_weight * adv_loss,
                f"id_loss_{flag}": self.id_loss_weight * id_loss,
                }

class DiscriminatorLoss(nn.Module):
    def __init__(self, disc_loss_weight = 1.0):
        super(DiscriminatorLoss, self).__init__()
        self.disc_loss_f = nn.MSELoss()  
        self.disc_loss_weight = disc_loss_weight
        
    def forward(self, fake_img, real_img, discriminator, flag):
        fake_score = discriminator(fake_img.detach())
        real_score = discriminator(real_img)
        disc_loss = 0.5 * (self.disc_loss_f(fake_score, torch.zeros_like(fake_score)) + self.disc_loss_f(real_score, torch.ones_like(real_score)))
        return {f"disc_loss_{flag}": self.disc_loss_weight * disc_loss}

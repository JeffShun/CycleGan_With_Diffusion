import warnings
warnings.filterwarnings('ignore')
import os
from config.model_config import network_cfg
import torch
from torch import optim
from torch.utils.data import DataLoader
import time
import itertools
from torch.autograd import Variable
from custom.utils.logger import Logger
from custom.utils.model_backup import model_backup
from custom.utils.lr_scheduler import WarmupMultiStepLR
from custom.utils.tensorboad_utils import get_writer
import random

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

def train():
    """创建日志与检查点目录""" 
    os.makedirs(network_cfg.gen_checkpoints_dir_A2B, exist_ok=True)
    os.makedirs(network_cfg.gen_checkpoints_dir_B2A, exist_ok=True)
    os.makedirs(network_cfg.disc_checkpoints_dir_A, exist_ok=True)
    os.makedirs(network_cfg.disc_checkpoints_dir_B, exist_ok=True)
    logger_dir = network_cfg.log_dir
    os.makedirs(logger_dir, exist_ok=True)
    logger = Logger(logger_dir + "/trainlog.txt", level='debug').logger
    tensorboad_dir = logger_dir + "/tf_logs"
    writer = get_writer(tensorboad_dir)

    """保存代码备份""" 
    model_backup(logger_dir + "/backup.tar")

    """构建网络并加载预训练"""
    GenNetA2B = network_cfg.gen_networkA2B.cuda()
    GenNetB2A = network_cfg.gen_networkB2A.cuda()   
    DiscNetA = network_cfg.disc_networkA.cuda()
    DiscNetB = network_cfg.disc_networkB.cuda()

    if os.path.exists(network_cfg.gen_load_from_A2B):
        print(f"Load pretrained Gen from {network_cfg.gen_load_from_A2B}")
        GenNetA2B.load_state_dict(torch.load(network_cfg.gen_load_from_A2B, map_location=network_cfg.device))
    if os.path.exists(network_cfg.gen_load_from_B2A):
        print(f"Load pretrained Gen from {network_cfg.gen_load_from_B2A}")
        GenNetB2A.load_state_dict(torch.load(network_cfg.gen_load_from_B2A, map_location=network_cfg.device))
    if os.path.exists(network_cfg.disc_load_from_A):
        print(f"Load pretrained Disc from {network_cfg.disc_load_from_A}")
        DiscNetA.load_state_dict(torch.load(network_cfg.disc_load_from_A, map_location=network_cfg.device))
    if os.path.exists(network_cfg.disc_load_from_B):
        print(f"Load pretrained Disc from {network_cfg.disc_load_from_B}")
        DiscNetB.load_state_dict(torch.load(network_cfg.disc_load_from_B, map_location=network_cfg.device))

    """损失函数 & 优化器 & 调度器 & 数据加载器""" 
    train_gen_loss_f = network_cfg.train_gen_loss_f
    train_disc_loss_f = network_cfg.train_disc_loss_f

    optimizer_gen = optim.Adam(
        itertools.chain(GenNetA2B.parameters(), GenNetB2A.parameters()), lr=network_cfg.lr_gen, betas=network_cfg.betas)
    optimizer_disc = optim.Adam(
        itertools.chain(DiscNetA.parameters(), DiscNetB.parameters()), lr=network_cfg.lr_disc, betas=network_cfg.betas)

    scheduler_gen = WarmupMultiStepLR(optimizer=optimizer_gen,
                                      milestones=network_cfg.milestones,
                                      gamma=network_cfg.gamma,
                                      warmup_factor=network_cfg.warmup_factor,
                                      warmup_iters=network_cfg.warmup_iters,
                                      warmup_method=network_cfg.warmup_method,
                                      last_epoch=network_cfg.last_epoch)
    scheduler_disc = WarmupMultiStepLR(optimizer=optimizer_disc,
                                       milestones=network_cfg.milestones,
                                       gamma=network_cfg.gamma,
                                       warmup_factor=network_cfg.warmup_factor,
                                       warmup_iters=network_cfg.warmup_iters,
                                       warmup_method=network_cfg.warmup_method,
                                       last_epoch=network_cfg.last_epoch)

    train_datasetA = network_cfg.train_datasetA
    train_dataloaderA = DataLoader(train_datasetA,
                                  batch_size=network_cfg.batchsize,
                                  shuffle=network_cfg.shuffle,
                                  num_workers=network_cfg.num_workers,
                                  drop_last=network_cfg.drop_last)
    train_datasetB = network_cfg.train_datasetB
    train_dataloaderB = DataLoader(train_datasetB,
                                  batch_size=network_cfg.batchsize,
                                  shuffle=network_cfg.shuffle,
                                  num_workers=network_cfg.num_workers,
                                  drop_last=network_cfg.drop_last)
    dataloader_len = min(len(train_dataloaderA), len(train_dataloaderB))
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()


    """训练主循环"""
    time_start = time.time()
    for epoch in range(network_cfg.start_epoch, network_cfg.total_epochs):
        GenNetA2B.train() 
        GenNetB2A.train()
        DiscNetA.train()
        DiscNetB.train()
        for ii, (imgA, imgB) in enumerate(zip(train_dataloaderA, train_dataloaderB)):
            imgA = Variable(imgA.float()).cuda()
            imgB = Variable(imgB.float()).cuda() 

            disc_loss_dict = dict()
            gen_loss_dict = dict()
            
            B = imgB.size(0)    
            
            # identity
            t_idB = GenNetA2B.sample_timesteps(B).cuda()
            noise_input_B, _ = GenNetA2B.noise_images(imgB, t_idB)
            idB = GenNetA2B(noise_input_B, imgB, t_idB)
            
            t_idA = GenNetB2A.sample_timesteps(B).cuda()
            noise_input_A, _ = GenNetB2A.noise_images(imgA, t_idA)
            idA = GenNetB2A(noise_input_A, imgA, t_idA)     

            # ABA cycle step
            t_A2B = torch.full((B,), GenNetA2B.noise_steps - 1, dtype=torch.long).cuda()
            noise_input = torch.randn_like(imgA)
            fakeB = GenNetA2B(noise_input, imgA, t_A2B)

            t_B2A = GenNetB2A.sample_timesteps(B).cuda()
            noised_imgA, _ = GenNetB2A.noise_images(imgA, t_B2A)
            recA = GenNetB2A(noised_imgA, fakeB, t_B2A)
            
            fakeB_pop = fake_B_buffer.push_and_pop(fakeB)
            disc_loss_B = train_disc_loss_f(fakeB_pop, imgB, DiscNetB, flag="B")
            disc_loss_dict.update(disc_loss_B)
            gen_loss_ABA = train_gen_loss_f(fakeB, imgA, idA, recA, DiscNetB, flag="B")
            gen_loss_dict.update(gen_loss_ABA)  

            # BAB cycle step
            t_B2A = torch.full((B,), GenNetB2A.noise_steps - 1, dtype=torch.long).cuda()
            noise_input = torch.randn_like(imgB)
            fakeA = GenNetB2A(noise_input, imgB, t_B2A)

            t_A2B = GenNetA2B.sample_timesteps(B).cuda()
            noised_imgB, _ = GenNetA2B.noise_images(imgB, t_A2B)
            recB = GenNetA2B(noised_imgB, fakeA, t_A2B)

            fakeA_pop = fake_A_buffer.push_and_pop(fakeA)
            disc_loss_A = train_disc_loss_f(fakeA_pop, imgA, DiscNetA, flag="A")
            disc_loss_dict.update(disc_loss_A)
            gen_loss_BAB = train_gen_loss_f(fakeA, imgB, idB, recB, DiscNetA, flag="A")
            gen_loss_dict.update(gen_loss_BAB)
            
            total_gen_loss = sum(gen_loss_ABA.values()) + sum(gen_loss_BAB.values())
            total_disc_loss = sum(disc_loss_A.values()) + sum(disc_loss_B.values())
            
            # === 优化器更新 ===
            optimizer_disc.zero_grad()
            total_disc_loss.backward()
            optimizer_disc.step()

            optimizer_gen.zero_grad()
            total_gen_loss.backward()
            optimizer_gen.step()

            """保存中间结果"""      
            realA = imgA[0:1].detach().cpu()
            fakeA = fakeA[0:1].detach().cpu()
            recA = recA[0:1].detach().cpu()
            realB = imgB[0:1].detach().cpu()
            fakeB = fakeB[0:1].detach().cpu()
            recB = recB[0:1].detach().cpu()

            concatA = torch.cat([realA, fakeB, recA], dim=3) * 0.5 + 0.5
            concatB = torch.cat([realB, fakeA, recB], dim=3) * 0.5 + 0.5

            writer.add_image('Image/A_Real_Fake_Rec', concatA[0], epoch + 1)  
            writer.add_image('Image/B_Real_Fake_Rec', concatB[0], epoch + 1)

            """写入日志"""
            for k, v in disc_loss_dict.items():
                writer.add_scalar(f'Loss/Discriminator/{k}', v.item(), epoch*dataloader_len+ii+1)

            for k, v in gen_loss_dict.items():
                writer.add_scalar(f'Loss/Generator/{k}', v.item(), epoch*dataloader_len+ii+1)

            progress = epoch + (ii + 1) / dataloader_len
            eta_min = ((network_cfg.total_epochs - progress) * (time.time() - time_start)) / 60 / progress
            eta_str = "{:.1f}min".format(eta_min) if eta_min < 60 else "{:.1f}h".format(eta_min / 60)

            disc_loss_str = " ".join(["{}={:.4f}".format(k, v.item()) for k, v in disc_loss_dict.items()])
            gen_loss_str = " ".join(["{}={:.4f}".format(k, v.item()) for k, v in gen_loss_dict.items()])

            logger.info("Epoch[{}/{}] Iter[{}/{}] Eta:{} | D({}) G({})".format(
                epoch + 1, network_cfg.total_epochs,
                ii + 1, dataloader_len,
                eta_str,
                disc_loss_str,
                gen_loss_str
            ))

        writer.add_scalar('LR', optimizer_gen.state_dict()['param_groups'][0]['lr'], epoch)
        scheduler_gen.step() 
        scheduler_disc.step()

        """保存模型"""
        gen_ckpt_pathA2B = os.path.join(network_cfg.gen_checkpoints_dir_A2B, f"{epoch + 1}.pth")
        gen_ckpt_pathB2A = os.path.join(network_cfg.gen_checkpoints_dir_B2A, f"{epoch + 1}.pth")
        disc_ckpt_pathA = os.path.join(network_cfg.disc_checkpoints_dir_A, f"{epoch + 1}.pth")
        disc_ckpt_pathB = os.path.join(network_cfg.disc_checkpoints_dir_B, f"{epoch + 1}.pth")
        torch.save(GenNetA2B.state_dict(), gen_ckpt_pathA2B)
        torch.save(GenNetB2A.state_dict(), gen_ckpt_pathB2A)
        torch.save(DiscNetA.state_dict(), disc_ckpt_pathA)
        torch.save(DiscNetB.state_dict(), disc_ckpt_pathB)

    writer.close()


if __name__ == '__main__':
	train()

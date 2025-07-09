import warnings
warnings.filterwarnings('ignore')
import os
import shutil
import time
import torch
import torch.distributed as dist
import itertools
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from config.model_config import network_cfg
from custom.utils.logger import Logger
from custom.utils.model_backup import model_backup
from custom.utils.lr_scheduler import WarmupMultiStepLR
from custom.utils.tensorboad_utils import get_writer
from custom.utils.dataloaderX import DataLoaderX
from custom.utils.distributed_utils import init_distributed_mode, cleanup
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
    """初始化分布式环境"""
    init_distributed_mode(network_cfg)
    rank = network_cfg.rank
    device = network_cfg.device
    is_main = (rank == 0)

    """创建日志与检查点目录""" 
    os.makedirs(network_cfg.gen_checkpoints_dir_A2B, exist_ok=True)
    os.makedirs(network_cfg.gen_checkpoints_dir_B2A, exist_ok=True)
    os.makedirs(network_cfg.disc_checkpoints_dir_A, exist_ok=True)
    os.makedirs(network_cfg.disc_checkpoints_dir_B, exist_ok=True)
    logger_dir = network_cfg.log_dir
    os.makedirs(logger_dir, exist_ok=True)
    logger = Logger(os.path.join(logger_dir, "trainlog.txt"), level='debug').logger
    tensorboard_dir = os.path.join(logger_dir, "tf_logs")
    writer = get_writer(tensorboard_dir)

    """保存代码备份""" 
    if is_main:
        model_backup(os.path.join(logger_dir, "backup.tar"))

    """构建网络并加载预训练"""
    GenNetA2B = network_cfg.gen_networkA2B.to(device)
    GenNetB2A = network_cfg.gen_networkB2A.to(device)
    DiscNetA = network_cfg.disc_networkA.to(device)
    DiscNetB = network_cfg.disc_networkB.to(device)

    gen_init_weightA2B = os.path.join(network_cfg.gen_checkpoints_dir_A2B, "initial_weightsA2B.pth")
    gen_init_weightB2A = os.path.join(network_cfg.gen_checkpoints_dir_B2A, "initial_weightsB2A.pth")
    disc_init_weightA = os.path.join(network_cfg.disc_checkpoints_dir_A, "initial_weightsA.pth")
    disc_init_weightB = os.path.join(network_cfg.disc_checkpoints_dir_B, "initial_weightsB.pth")

    if os.path.exists(network_cfg.gen_load_from_A2B):
        if is_main: print(f"Load pretrained Gen from {network_cfg.gen_load_from_A2B}")
        GenNetA2B.load_state_dict(torch.load(network_cfg.gen_load_from_A2B, map_location=device))
    if os.path.exists(network_cfg.gen_load_from_B2A):
        if is_main: print(f"Load pretrained Gen from {network_cfg.gen_load_from_B2A}")
        GenNetB2A.load_state_dict(torch.load(network_cfg.gen_load_from_B2A, map_location=device))
    if os.path.exists(network_cfg.disc_load_from_A):
        if is_main: print(f"Load pretrained Disc from {network_cfg.disc_load_from_A}")
        DiscNetA.load_state_dict(torch.load(network_cfg.disc_load_from_A, map_location=device))
    if os.path.exists(network_cfg.disc_load_from_B):
        if is_main: print(f"Load pretrained Disc from {network_cfg.disc_load_from_B}")
        DiscNetB.load_state_dict(torch.load(network_cfg.disc_load_from_B, map_location=device))

    if not os.path.exists(network_cfg.gen_load_from_A2B) or not os.path.exists(network_cfg.gen_load_from_B2A)\
        or not os.path.exists(network_cfg.disc_load_from_A) or not os.path.exists(network_cfg.disc_load_from_B):
        # 如果没有指定从特定路径加载，则尝试加载初始权重
        if is_main:
            torch.save(GenNetA2B.state_dict(), gen_init_weightA2B)
            torch.save(GenNetB2A.state_dict(), gen_init_weightB2A)
            torch.save(DiscNetA.state_dict(), disc_init_weightA)
            torch.save(DiscNetB.state_dict(), disc_init_weightB)
        dist.barrier()
        print(f"Load initial Gen weights from {gen_init_weightA2B}")
        GenNetA2B.load_state_dict(torch.load(gen_init_weightA2B, map_location=device))
        print(f"Load initial Gen weights from {gen_init_weightB2A}")
        GenNetB2A.load_state_dict(torch.load(gen_init_weightB2A, map_location=device))
        print(f"Load initial Disc weights from {disc_init_weightA}")
        DiscNetA.load_state_dict(torch.load(disc_init_weightA, map_location=device))
        print(f"Load initial Disc weights from {disc_init_weightB}")
        DiscNetB.load_state_dict(torch.load(disc_init_weightB, map_location=device))

    """SyncBN + DDP""" 
    GenNetA2B = torch.nn.SyncBatchNorm.convert_sync_batchnorm(GenNetA2B)
    GenNetB2A = torch.nn.SyncBatchNorm.convert_sync_batchnorm(GenNetB2A)
    DiscNetA = torch.nn.SyncBatchNorm.convert_sync_batchnorm(DiscNetA)
    DiscNetB = torch.nn.SyncBatchNorm.convert_sync_batchnorm(DiscNetB)
    
    GenNetA2B = torch.nn.parallel.DistributedDataParallel(GenNetA2B, device_ids=[network_cfg.gpu], find_unused_parameters=True)
    GenNetB2A = torch.nn.parallel.DistributedDataParallel(GenNetB2A, device_ids=[network_cfg.gpu], find_unused_parameters=True)
    DiscNetA = torch.nn.parallel.DistributedDataParallel(DiscNetA, device_ids=[network_cfg.gpu], find_unused_parameters=True)
    DiscNetB = torch.nn.parallel.DistributedDataParallel(DiscNetB, device_ids=[network_cfg.gpu], find_unused_parameters=True)

    """损失函数 & 优化器 & 调度器 & 数据加载器""" 
    train_gen_loss_f = network_cfg.train_gen_loss_f
    train_disc_loss_f = network_cfg.train_disc_loss_f
    
    optimizer_gen = optim.Adam(
        itertools.chain(GenNetA2B.parameters(), GenNetB2A.parameters()), lr=network_cfg.lr_gen, betas=network_cfg.betas)
    optimizer_disc = optim.Adam(
        itertools.chain(DiscNetA.parameters(), DiscNetB.parameters()), lr=network_cfg.lr_disc, betas=network_cfg.betas)

    scheduler_gen = WarmupMultiStepLR(
        optimizer_gen,
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

    train_samplerA = torch.utils.data.distributed.DistributedSampler(network_cfg.train_datasetA, shuffle=network_cfg.shuffle)
    train_samplerB = torch.utils.data.distributed.DistributedSampler(network_cfg.train_datasetB, shuffle=network_cfg.shuffle)

    train_dataloaderA = DataLoader(
        dataset=network_cfg.train_datasetA,
        batch_size=network_cfg.batchsize,
        sampler=train_samplerA,
        num_workers=network_cfg.num_workers,
        drop_last=network_cfg.drop_last,
        pin_memory=False)

    train_dataloaderB = DataLoader(
        dataset=network_cfg.train_datasetB,
        batch_size=network_cfg.batchsize,
        sampler=train_samplerB,
        num_workers=network_cfg.num_workers,
        drop_last=network_cfg.drop_last,
        pin_memory=False)
    
    dataloader_len = min(len(train_dataloaderA), len(train_dataloaderB))
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    """训练主循环"""
    time_start = time.time()
    for epoch in range(network_cfg.start_epoch, network_cfg.total_epochs):
        train_samplerA.set_epoch(epoch)
        train_samplerB.set_epoch(epoch)
        GenNetA2B.train() 
        GenNetB2A.train()
        DiscNetA.train()
        DiscNetB.train()
        for ii, (imgA, imgB) in enumerate(zip(train_dataloaderA, train_dataloaderB)):
            imgA = Variable(imgA.float()).to(device)
            imgB = Variable(imgB.float()).to(device)

            disc_loss_dict = dict()
            gen_loss_dict = dict()
            
            B = imgB.size(0)

            # identity
            t_idB = GenNetA2B.module.sample_timesteps(B).to(device)
            noise_input_B, _ = GenNetA2B.module.noise_images(imgB, t_idB)
            idB = GenNetA2B(noise_input_B, imgB, t_idB)
            
            t_idA = GenNetB2A.module.sample_timesteps(B).to(device)
            noise_input_A, _ = GenNetB2A.module.noise_images(imgA, t_idA)
            idA = GenNetB2A(noise_input_A, imgA, t_idA)        

            # ABA cycle step
            t_A2B = torch.full((B,), GenNetA2B.module.noise_steps - 1, dtype=torch.long).to(device)
            noise_inputB = torch.randn_like(imgB)
            fakeB = GenNetA2B(noise_inputB, imgA, t_A2B)

            t_B2A = GenNetB2A.module.sample_timesteps(B).to(device)
            noised_imgA, _ = GenNetB2A.module.noise_images(imgA, t_B2A)
            recA = GenNetB2A(noised_imgA, fakeB, t_B2A)

            fakeB_pop = fake_B_buffer.push_and_pop(fakeB)
            disc_loss_B = train_disc_loss_f(fakeB_pop, imgB, DiscNetB, flag="B")
            disc_loss_dict.update(disc_loss_B)
            gen_loss_ABA = train_gen_loss_f(fakeB, imgA, idA, recA, DiscNetB, flag="B")
            gen_loss_dict.update(gen_loss_ABA)  

            # BAB cycle step
            t_B2A = torch.full((B,), GenNetB2A.module.noise_steps - 1, dtype=torch.long).to(device)
            noise_inputA = torch.randn_like(imgA)
            fakeA = GenNetB2A(noise_inputA, imgB, t_B2A)

            t_A2B = GenNetA2B.module.sample_timesteps(B).to(device)
            noised_imgB, _ = GenNetA2B.module.noise_images(imgB, t_A2B)
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

            if is_main:
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
    
        if is_main:        
            writer.add_scalar('LR', optimizer_gen.state_dict()['param_groups'][0]['lr'], epoch)

        scheduler_gen.step() 
        scheduler_gen.step() 
        scheduler_disc.step()

        if is_main:
            """保存模型"""
            gen_ckpt_pathA2B = os.path.join(network_cfg.gen_checkpoints_dir_A2B, f"{epoch + 1}.pth")
            gen_ckpt_pathB2A = os.path.join(network_cfg.gen_checkpoints_dir_B2A, f"{epoch + 1}.pth")
            disc_ckpt_pathA = os.path.join(network_cfg.disc_checkpoints_dir_A, f"{epoch + 1}.pth")
            disc_ckpt_pathB = os.path.join(network_cfg.disc_checkpoints_dir_B, f"{epoch + 1}.pth")
            torch.save(GenNetA2B.module.state_dict(), gen_ckpt_pathA2B)
            torch.save(GenNetB2A.module.state_dict(), gen_ckpt_pathB2A)
            torch.save(DiscNetA.module.state_dict(), disc_ckpt_pathA)
            torch.save(DiscNetB.module.state_dict(), disc_ckpt_pathB)

    if is_main: 
        writer.close()
    dist.barrier()
    cleanup()


if __name__ == '__main__':
    train()
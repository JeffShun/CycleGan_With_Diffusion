import sys, os
work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(work_dir)
from custom.dataset.dataset import MyDataset
from custom.utils.data_transforms import *
from custom.model.backbones.UNet import UNet
from custom.model.generator import DiffusionModel
from custom.model.discriminator import GlobalDiscriminator, PatchDiscriminator
from custom.model.model_loss import GeneratorLoss, DiscriminatorLoss

class network_cfg:
    # img
    img_size = (320, 320)

    # network
    gen_networkA2B = DiffusionModel(
        backbone = UNet(
            c_in=2, 
            c_out=1,
            base_channel=32,
            attention_heads=8,
            time_dim=256
            ),
        img_size=img_size,
        noise_steps=1000, 
        beta_schedule="linear"
        )

    gen_networkB2A = DiffusionModel(
        backbone = UNet(
            c_in=2, 
            c_out=1,
            base_channel=32,
            attention_heads=8,
            time_dim=256
            ),
        img_size=img_size,
        noise_steps=1000, 
        beta_schedule="linear"
        )

    disc_networkA = PatchDiscriminator(
        in_channels = 1,
        base_channels = 64,
        num_layers = 4
        )
    
    disc_networkB = PatchDiscriminator(
        in_channels = 1,
        base_channels = 64,
        num_layers = 4
        )  

    # loss function
    train_gen_loss_f = GeneratorLoss(consist_loss_weight=10, adv_loss_weight=1, id_loss_weight=5)
    train_disc_loss_f = DiscriminatorLoss(disc_loss_weight=1)

    # dataset
    train_datasetA = MyDataset(
        dst_list_file = work_dir + "/train_data/processed_data/A/dataA.txt",
        transforms = TransformCompose([
            to_tensor(),
            normlize(win_clip=None),
            # random_flip(axis=1, prob=0.5),
            # random_flip(axis=2, prob=0.5),
            # random_rotate90(k=1, prob=0.5),
            resize(img_size),
            ])
        )
    
    train_datasetB = MyDataset(
        dst_list_file = work_dir + "/train_data/processed_data/B/dataB.txt",
        transforms = TransformCompose([
            to_tensor(),
            normlize(win_clip=None),
            # random_flip(axis=1, prob=0.5),
            # random_flip(axis=2, prob=0.5),
            # random_rotate90(k=1, prob=0.5),
            resize(img_size)
            ])
        )

    # dataloader
    batchsize = 4
    shuffle = True
    num_workers = 4
    drop_last = True
    # optimizer
    lr_gen = 5e-5
    lr_disc = 5e-5
    betas = (0.5, 0.999)

    # scheduler
    milestones = [50, 100, 150]
    gamma = 0.5
    warmup_factor = 0.1
    warmup_iters = 0
    warmup_method = "linear"
    last_epoch = -1

    # debug
    log_dir = work_dir + "/Logs"
    gen_checkpoints_dir_A2B = work_dir + '/checkpoints/generator/A2B'
    gen_checkpoints_dir_B2A = work_dir + '/checkpoints/generator/B2A'
    disc_checkpoints_dir_A = work_dir + '/checkpoints/discriminator/A'
    disc_checkpoints_dir_B = work_dir + '/checkpoints/discriminator/B'

    checkpoint_save_interval = 1
    total_epochs = 200
    start_epoch = 2
    gen_load_from_A2B = work_dir + '/checkpoints/generator/A2B/2.pth'
    gen_load_from_B2A = work_dir + '/checkpoints/generator/B2A/2.pth'
    disc_load_from_A = work_dir + '/checkpoints/discriminator/A/2.pth'
    disc_load_from_B = work_dir + '/checkpoints/discriminator/B/2.pth'

    # others
    device = torch.device('cuda')
    dist_backend = 'nccl'
    dist_url = 'env://'

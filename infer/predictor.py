from calendar import c
from dis import dis
from os.path import abspath, dirname
from typing import IO, Dict
from torch.nn import functional as F
import numpy as np
import torch
import yaml

from train.config.model_config import network_cfg

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class DiffusionConfig:

    def __init__(self, test_cfg):
        # 配置文件
        self.img_size = test_cfg.get("img_size")

    def __repr__(self) -> str:
        return str(self.__dict__)


class DiffusionModel:

    def __init__(self, model_f: IO, config_f):
        # TODO: 模型文件定制
        self.model_f = model_f 
        self.config_f = config_f
        self.network_cfg = network_cfg


class DiffusionPredictor:

    def __init__(self, device: str, model: DiffusionModel):
        self.device = torch.device(device)
        self.model = model

        with open(self.model.config_f, 'r') as config_f:
            self.test_cfg = DiffusionConfig(yaml.safe_load(config_f))
        self.network_cfg = model.network_cfg
        self.load_model()

    def load_model(self) -> None:
        if isinstance(self.model.model_f, str):
            # 根据后缀判断类型
            if self.model.model_f.endswith('.pth'):
                self.load_model_pth()

    def load_model_pth(self) -> None:
        # 加载动态图
        self.net = self.network_cfg.gen_networkB2A
        checkpoint = torch.load(self.model.model_f, map_location=self.device)
        self.net.load_state_dict(checkpoint)
        self.net.eval()
        self.net.to(self.device)

    def predict(self, img):
        # pytorch预测
        with torch.no_grad():
            img = ((img - img.min()) / (img.max() - img.min()) - 0.5) / 0.5
            img_t = torch.from_numpy(img.astype(np.float32))[None, None] 
            img_t = F.interpolate(img_t, size=self.test_cfg.img_size, mode="bilinear")
            output = self.net.ddim_sample(img_t.to(self.device), ddim_timesteps=5)
            output = output.squeeze().cpu().detach().numpy()      
        return output

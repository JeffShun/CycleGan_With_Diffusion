import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity
import cv2
import matplotlib.font_manager as fm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infer.predictor import DiffusionModel, DiffusionPredictor
from scipy.ndimage import zoom
import SimpleITK as sitk

def parse_args():
    parser = argparse.ArgumentParser(description='Test MRI Reconstruction')

    parser.add_argument('--device', default="cuda:2", type=str)
    parser.add_argument('--input_path', default='./data/input/t2', type=str)
    parser.add_argument('--output_path', default='./data/output', type=str)

    parser.add_argument(
        '--model_file',
        type=str,
        default='../train/checkpoints/generator/B2A/12.pth'
    )
    parser.add_argument(
        '--config_file',
        type=str,
        default='./test_config.yaml'
    )
    args = parser.parse_args()
    return args


def save_img(imgs, save_path, img_size=(768, 768)):
    img_os = []
    for img in imgs:
        img = (img-img.min())/(img.max()-img.min())
        img_o = zoom(img, img_size / np.array(img.shape), order=1)
        img_o *= 255
        img_o = np.clip(img_o, 0, 255)
        img_os.append(img_o)
    save_img = np.concatenate(img_os,1)
    save_img = Image.fromarray(save_img.astype(np.uint8))
    save_img.save(save_path)

def main(input_path, output_path, device, args):
    # TODO: 适配参数输入
    model_recon = DiffusionModel(
        model_f=args.model_file,
        config_f=args.config_file,
    )
    predictor_recon = DiffusionPredictor(
        device=device,
        model=model_recon,
    )

    os.makedirs(output_path, exist_ok=True)

    for pid in tqdm(os.listdir(input_path)):
        imgs_dir = os.path.join(input_path, pid)
        src_imgs = os.listdir(imgs_dir)
        for img in src_imgs:
            try:
                src_img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(imgs_dir, img)))[0]
                converted_img = predictor_recon.predict(src_img)
                save_img([src_img, converted_img], os.path.join(output_path, f'{img.replace(".dcm","")}.png'))
            except:
                raise Exception("Error in %s"%(img))

if __name__ == '__main__':
    args = parse_args()
    main(
        input_path=args.input_path,
        output_path=args.output_path,
        device=args.device,
        args=args,
    )


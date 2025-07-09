import argparse
import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from multiprocessing import Pool
import pydicom

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./train_data/origin_data')
    parser.add_argument('--save_path', type=str, default='./train_data/processed_data')
    return parser.parse_args()

def correctify_dcm(data_path):
    try:
        image = pydicom.dcmread(data_path)
        image.SpacingBetweenSlices = image.SliceThickness
        image.save_as(data_path)
    except Exception as e:
        print(f"Error correcting DICOM: {data_path} - {e}")

def read_dcm_image(path):
    try:
        img = sitk.ReadImage(path)
    except Exception:
        correctify_dcm(path)
        img = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(img)[0]

def get_sorted_dcm_list(folder):
    return sorted([
        os.path.join(root, f)
        for root, _, files in os.walk(folder)
        for f in files if f.endswith('.dcm')
    ])

def process_single(input_tuple):
    dcm_path, save_path, subdir = input_tuple
    name = os.path.splitext(os.path.basename(dcm_path))[0]
    img = read_dcm_image(dcm_path)
    np.save(os.path.join(save_path, subdir, name + '.npy'), img)

def gen_lst(save_path):
    A_dir = os.path.join(save_path, 'A')
    B_dir = os.path.join(save_path, 'B')

    A_list_file = os.path.join(A_dir, 'dataA.txt')
    B_list_file = os.path.join(B_dir, 'dataB.txt')

    A_files = sorted([f for f in os.listdir(A_dir) if f.endswith('.npy')])
    B_files = sorted([f for f in os.listdir(B_dir) if f.endswith('.npy')])

    with open(A_list_file, 'w') as f:
        for name in A_files:
            f.write(os.path.join(save_path, 'A', name).replace("\\", "/") + '\n')

    with open(B_list_file, 'w') as f:
        for name in B_files:
            f.write(os.path.join(save_path, 'B', name).replace("\\", "/") + '\n')


def main():
    args = parse_args()
    os.makedirs(os.path.join(args.save_path, 'A'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'B'), exist_ok=True)

    t1_dir = os.path.join(args.data_path, "Head_t1_flair_tra")
    t2_dir = os.path.join(args.data_path, "Head_t2_fse_tra")

    t1_list = get_sorted_dcm_list(t1_dir)
    t2_list = get_sorted_dcm_list(t2_dir)

    print(f"Saving {len(t1_list)} T1 images and {len(t2_list)} T2 images...")

    with Pool(8) as pool:
        inputs_A = [(p, args.save_path, 'A') for p in t1_list]
        inputs_B = [(p, args.save_path, 'B') for p in t2_list]
        list(tqdm(pool.imap_unordered(process_single, inputs_A), total=len(inputs_A)))
        list(tqdm(pool.imap_unordered(process_single, inputs_B), total=len(inputs_B)))

    gen_lst(args.save_path)

if __name__ == '__main__':
    main()
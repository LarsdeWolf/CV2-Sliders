import os
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import argparse
from data_loader_loc import Flare_Image_Loader
from torchvision.utils import save_image
from pathlib import Path


class MyTranslationTransform(object):
    def __init__(self, position):
        self.position = position

    def __call__(self, x):
        return TF.affine(x, angle=0, scale=1, shear=[0, 0], translate=list(self.position))


def generate(
        amount: int = 1,
        strengths: list = None,
        flare_scatters: list = None,
        output_dir: str = 'C:/Assignments/Computer_Vision_2/Flare7K/datasets/test',
):
    if strengths is None:
        strengths = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    if flare_scatters is None:
        flare_scatters = os.listdir('dataset/Flare7k/Scattering_Flare')
    for strength in strengths:
        Path(f"{output_dir}\\{strength}").mkdir(parents=True, exist_ok=True)
    transform_base = transforms.Compose([transforms.RandomCrop((512, 512), pad_if_needed=True, padding_mode='reflect'),
                                         transforms.RandomHorizontalFlip()
                                         ])

    flare_image_loader = Flare_Image_Loader('dataset/Flickr24K', strengths, transform_base,
                                            transform_flare=None)
    base_indices = np.random.permutation(len(flare_image_loader.data_list))[:amount]
    for scatter in flare_scatters:
        flare_image_loader.load_scattering_flare('dataset/Flare7K', f'dataset/Flare7k/Scattering_Flare/{scatter}')
        flare_image_loader.load_reflective_flare('dataset/Flare7K', 'dataset/Flare7k/Reflective_Flare')

        for base in base_indices:
            images = flare_image_loader[base]
            for strength in strengths:
                base_img, flare_img, merge_img, _ = images[strength]
                save_image(merge_img, f"{output_dir}/{strength}/{base}_{scatter}.png")
                save_image(flare_img, f"{output_dir}/flares/{base}_{scatter}.png")


if __name__=="main":
    parser = argparse.ArgumentParser()
    parser.add_argument('amount', type=int,
                        help='amount of image pairs to generate')
    parser.add_argument('strengths', type=list,
                        help='list of floats indicating the flare strenghts')
    parser.add_argument('output_dir', type=str,
                        help='output directory of the created flare dataset')
    args = parser.parse_args()
    generate(args.amount, args.strengths, [], args.output_dir)

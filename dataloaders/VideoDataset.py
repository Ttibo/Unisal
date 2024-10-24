from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import cv2
import PIL
import random
import os
import re
import glob

decimal_pattern = re.compile(r'(\d+)\.jpg$')
def extract_decimal(filename):
    match = decimal_pattern.search(filename)
    if match:
        return float(match.group(1))  # Convertir le nombre en float
    return float('inf')  # Si aucun nombre trouvé, renvoyer une valeur très élevée pour les placer à la fin


def normalize_tensor(tensor, rescale=False):
    tmin = torch.min(tensor)
    if rescale or tmin < 0:
        tensor -= tmin
    tsum = tensor.sum()
    return tensor / tsum if tsum > 0 else tensor.fill_(1. / tensor.numel())

class VideoDataset(Dataset):
    def __init__(
            self,
            path,
            preproc_cfg=None,
            seq_len=12,
            frame_modulo=5,
            phase='train',
            extension = "jpg",
            fix_dir = "fixation",
            sal_dir = "maps",
            img_dir = "images",
            ratio_val_test = 0.858,
            limit= None
            ):
        self.ratio_val_test = ratio_val_test 
        self.fix_dir = fix_dir
        self.sal_dir = sal_dir
        self.img_dir = img_dir
        self.limit = limit

        self.preproc_cfg = {
            'rgb_mean': (0.485, 0.456, 0.406),
            'rgb_std': (0.229, 0.224, 0.225),
        }
        if preproc_cfg is not None:
            self.preproc_cfg.update(preproc_cfg)
        self.dir = Path(path)
        self.seq_len = seq_len
        self.frame_modulo = frame_modulo
        self.phase = phase
        self.extension = extension
        self.load_data()

    def load_data(self):
        self.all_video_folders = []
        for folder_ in sorted(self.dir.iterdir()):

            # Parcours de chaque vidéo
            if folder_.stem == ".DS_Store":
                continue

            self.all_video_folders.append({
                'img' : os.path.join(self.dir, folder_.stem, self.img_dir), 
                'fix' : os.path.join(self.dir, folder_.stem, self.fix_dir), 
                'sal' : os.path.join(self.dir, folder_.stem, self.sal_dir),
                'len' : len(list(Path(os.path.join(self.dir, folder_.stem, self.img_dir)).glob(f"*.{self.extension}") ))
            })

            if self.all_video_folders[-1]['len'] - (self.seq_len - 1) * (self.frame_modulo + 1) < 1 : 
                self.all_video_folders.pop(-1)

        if self.limit is not None:
            self.all_video_folders = self.all_video_folders[:min(self.limit , len(self.all_video_folders))]

        # separate btw train and val 
        if self.phase == "train":
            self.all_video_folders = self.all_video_folders[0:int(len(self.all_video_folders)*self.ratio_val_test)]
        elif self.phase == "val":
            self.all_video_folders = self.all_video_folders[int(len(self.all_video_folders)*self.ratio_val_test):]

    def load_frames(self, folder):
        
        frames = sorted((Path(folder['img'])).glob(f"*.{self.extension}"), key=lambda f: extract_decimal(f.name))
        saliency_maps = sorted((Path(folder['sal'])).glob(f"*.{self.extension}"), key=lambda f: extract_decimal(f.name))
        fixation_maps = sorted((Path(folder['fix'])).glob(f"*.{self.extension}"), key=lambda f: extract_decimal(f.name))

        return frames, saliency_maps, fixation_maps

    def preprocess(self, img, out_size=None, data='img'):
        transformations = [
            transforms.ToPILImage(),
            transforms.Resize(out_size, interpolation=PIL.Image.LANCZOS if data != 'fix' else PIL.Image.NEAREST),
            transforms.ToTensor()
        ]
        if data == 'img':
            transformations.append(transforms.Normalize(self.preproc_cfg['rgb_mean'], self.preproc_cfg['rgb_std']))
        elif data == 'sal':
            transformations.append(transforms.Lambda(normalize_tensor))
        elif data == 'fix':
            transformations.append(transforms.Lambda(lambda fix: torch.gt(fix, 0.1)))
        return transforms.Compose(transformations)(img)

    def get_out_size(self, img_size, train=True):
        if train:
            selection = [(8, 13), (9, 13), (9, 12), (12, 9), (13, 9)]
        else:
            selection = [(n1, n2) for n1 in range(7, 14) for n2 in range(7, 14) if 100 <= n1 * n2 <= 120]
       
        ar = img_size[0] / img_size[1]
        best_size = max(selection, key=lambda s: min(ar, s[0] / s[1]) / max(ar, s[0] / s[1]))
        return  (288, 384)
        return tuple(r * 32 for r in best_size)

    def get_frame_nrs(self,vid):
        max_start_index = vid['len'] - (self.seq_len - 1) * (self.frame_modulo + 1)
        if max_start_index <= 0:
            raise ValueError("Impossible de sélectionner les indices avec les paramètres donnés.")

        # Choisir un index de départ aléatoire
        start_index = random.randint(0, max_start_index - 1)
        
        # Générer les indices à partir de l'index de départ
        indices = [start_index + i * (self.frame_modulo + 1) for i in range(self.seq_len)]
        return indices 
 
    def __getitem__(self, idx):
        folder = self.all_video_folders[idx]
        frames, saliency_maps, fixation_maps = self.load_frames(folder)
        frame_indices = self.get_frame_nrs(folder)

        video_frames = []
        saliency_frames = []
        fixation_frames = []

        for i in frame_indices:
            img = cv2.imread(str(frames[i]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = self.preprocess(img, out_size=self.get_out_size(img.shape), data='img')

            sal = cv2.imread(str(saliency_maps[i]), cv2.IMREAD_GRAYSCALE)
            sal_tensor = self.preprocess(sal, out_size=self.get_out_size(img.shape), data='sal')

            fix = cv2.imread(str(fixation_maps[i]), cv2.IMREAD_GRAYSCALE)
            fix_tensor = self.preprocess(fix, out_size=self.get_out_size(img.shape), data='fix')

            video_frames.append(img_tensor)
            saliency_frames.append(sal_tensor)
            fixation_frames.append(fix_tensor)

        # Empiler les frames sélectionnées pour former un tenseur [N, C, W, H]
        video_tensor = torch.stack(video_frames)  # [N, C, W, H]
        saliency_tensor = torch.stack(saliency_frames)  # [N, 1, W, H]
        fixation_tensor = torch.stack(fixation_frames)  # [N, 1, W, H]


        # print(f"Tensors video {video_tensor.shape} : saliency {saliency_tensor.shape} : fixation {fixation_tensor.shape}")
        return video_tensor , saliency_tensor , fixation_tensor, (224,224)

    def __len__(self):
        return len(self.all_video_folders)

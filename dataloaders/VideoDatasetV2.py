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

class VideoDatasetV2(Dataset):
    def __init__(self, path, N=4, skip = 2, verbose=1, preproc_cfg=None):
        self.verbose = verbose
        self.skip = skip
        self.preproc_cfg = {
            'rgb_mean': (0.485, 0.456, 0.406),
            'rgb_std': (0.229, 0.224, 0.225),
        }
        if preproc_cfg is not None:
            self.preproc_cfg.update(preproc_cfg)
        self.dir = Path(path)
        self.N = N  # Number of frames to sample randomly
        self.load_data()

        print("Path:", path)
        print(f"Numbers videos {len(self.all_video_folders)}")

    @property
    def fix_dir(self): return self.dir / 'fixations'
    @property
    def sal_dir(self): return self.dir / 'maps'
    @property
    def img_dir(self): return self.dir / 'frames'

    def load_data(self):
        self.all_video_folders = []
        for folder_ in sorted(self.img_dir.iterdir()):
            # Parcours de chaque vidéo
            if folder_.stem == ".DS_Store":
                continue
            self.all_video_folders.append({
                'img' : os.path.join(self.img_dir , folder_.stem + ".mp4"), 
                'fix' : os.path.join(self.fix_dir , folder_.stem + ".mp4"), 
                'sal' : os.path.join(self.sal_dir , folder_.stem + ".mp4")
            })


        # for each video load all frames
        self.all_frames_data = []
        for video in self.all_video_folders:
            # load frames
            frames_ , sals_ , fixs_ = self.load_frames(video)



            index = 0
            for fra , sal , fix in zip(frames_, sals_ , fixs_):
                self.all_frames_data.append({
                    'video' : video,
                    'len' : len(frames_),
                    'img' : fra,
                    'fix' : fix,
                    'sal' : sal,
                    'index' : index
                })

                index += 1


    def load_frames(self, folder):
        frames = sorted((Path(folder['img'])).glob("*.jpg"), key=lambda f: extract_decimal(f.name))
        saliency_maps = sorted((Path(folder['sal'])).glob("*.jpg"), key=lambda f: extract_decimal(f.name))
        fixation_maps = sorted((Path(folder['fix'])).glob("*.jpg"), key=lambda f: extract_decimal(f.name))

        fr, sl, fi = [] , [] , []
        count = 0
        for fra , sal , fix in zip(frames ,saliency_maps, fixation_maps):
            count += 1
            if count < self.skip:
                continue

            count = 0

            fr.append(fra)
            sl.append(sal)
            fi.append(fix)




        return fr, sl , fi

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

        return tuple(r * 32 for r in best_size)
 
    def __getitem__(self, idx):
        frame_data = self.all_frames_data[idx]

        # num_frames = len(frames)
        half_window = self.N // 2

        # Calculer les indices de début et de fin pour la fenêtre autour de idx
        start_index = max(0, frame_data['index'] - half_window)
        end_index = min(frame_data['len'], start_index + self.N)
        
        # Si on est proche de la fin de la vidéo, ajuster start_index pour toujours avoir N frames
        if end_index - start_index < self.N:
            start_index = max(0, end_index - self.N)

        frame_indices = list(range(start_index + (idx - frame_data['index']), end_index + (idx - frame_data['index'])))

        video_frames = []
        saliency_frames = []
        fixation_frames = []

        for i in frame_indices:
            img = cv2.imread(str(self.all_frames_data[i]['img']))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = self.preprocess(img, out_size=self.get_out_size(img.shape), data='img')

            sal = cv2.imread(str(self.all_frames_data[i]['sal']), cv2.IMREAD_GRAYSCALE)
            sal_tensor = self.preprocess(sal, out_size=self.get_out_size(img.shape), data='sal')

            fix = cv2.imread(str(self.all_frames_data[i]['fix']), cv2.IMREAD_GRAYSCALE)
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
        return len(self.all_frames_data)

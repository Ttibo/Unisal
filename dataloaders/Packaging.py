from pathlib import Path
import os
import json

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import cv2
import PIL

def normalize_tensor(tensor, rescale=False):
    tmin = torch.min(tensor)
    if rescale or tmin < 0:
        tensor -= tmin
    tsum = tensor.sum()
    if tsum > 0:
        return tensor / tsum
    print("Zero tensor")
    tensor.fill_(1. / tensor.numel())
    return tensor

class PACKAGINGDataset(Dataset):

    n_train_val_images = 64
    dynamic = False

    def __init__(self, path , phase='train', subset=None, verbose=1,
                 preproc_cfg=None, n_x_val=10, x_val_step=0, x_val_seed=27):
        self.phase = phase
        self.train = phase == 'train'
        self.subset = subset
        self.verbose = verbose
        self.preproc_cfg = {
            'rgb_mean': (0.485, 0.456, 0.406),
            'rgb_std': (0.229, 0.224, 0.225),
        }
        if preproc_cfg is not None:
            self.preproc_cfg.update(preproc_cfg)

        self.n_x_val = n_x_val
        self.x_val_step = x_val_step
        self.x_val_seed = x_val_seed
        self.dir = Path(path)

        # Cross-validation split
        n_images = self.n_train_val_images
        if x_val_step is None:
            self.samples = np.arange(0, n_images)
        else:
            print(f"X-Val step: {x_val_step}")
            assert(self.x_val_step < self.n_x_val)
            samples = np.arange(0, n_images)
            if self.x_val_seed > 0:
                np.random.seed(self.x_val_seed)
                np.random.shuffle(samples)
            val_start = int(len(samples) / self.n_x_val * self.x_val_step)
            val_end = int(len(samples) / self.n_x_val * (self.x_val_step + 1))
            samples = samples.tolist()
            if not self.train:
                self.samples = samples[val_start:val_end]
            else:
                del samples[val_start:val_end]
                self.samples = samples

        self.all_image_files, self.size_dict = self.load_data()
        if self.subset is not None:
            self.samples = self.samples[:int(len(self.samples) * subset)]
        # For compatibility with video datasets
        self.n_images_dict = {sample: 1 for sample in self.samples}
        self.target_size_dict = {
            img_idx: self.size_dict[img_idx]['target_size']
            for img_idx in self.samples}
        self.n_samples = len(self.samples)
        self.frame_modulo = 1

    def get_map(self, img_idx):
        """
        Récupère la carte de saillance pour une image donnée.
        
        Paramètres :
        - img_idx (int) : Index de l'image.
        
        Retourne :
        - map (ndarray) : Carte de saillance.
        """
        map_file = self.sal_dir / self.all_image_files[img_idx]['map']
        map = cv2.imread(str(map_file), cv2.IMREAD_GRAYSCALE)
        assert(map is not None)
        return map

    def get_img(self, img_idx):
        """
        Récupère l'image RGB pour un index donné.

        Paramètres :
        - img_idx (int) : Index de l'image.
        
        Retourne :
        - img (ndarray) : Image RGB.
        """
        img_file = self.img_dir / self.all_image_files[img_idx]['img']
        img = cv2.imread(str(img_file))
        assert(img is not None)
        return np.ascontiguousarray(img[:, :, ::-1])

    def get_fixation_map(self, img_idx):
        """
        Récupère la carte de fixation pour une image donnée.

        Paramètres :
        - img_idx (int) : Index de l'image.
        
        Retourne :
        - fix_map (ndarray) : Carte de fixation.
        """
        fix_map_file = self.fix_dir / self.all_image_files[img_idx]['pts']
        fix_map = cv2.imread(str(fix_map_file), cv2.IMREAD_GRAYSCALE)
        assert(fix_map is not None)
        return fix_map


    @property
    def fix_dir(self):
        # return self.dir / 'ALLFIXATIONMAPS' / 'ALLFIXATIONMAPS'
        return self.dir / 'fixations' 
    
    @property
    def sal_dir(self):
        # return self.dir / 'ALLFIXATIONMAPS' / 'ALLFIXATIONMAPS'
        return self.dir / 'maps' 
    

    @property
    def img_dir(self):
        # return self.dir / 'ALLSTIMULI' / 'ALLSTIMULI'
        return self.dir / 'images' 


    def get_out_size_eval(self, img_size):
        """
        Calcule la taille de sortie pour l'évaluation en fonction de la taille de l'image.

        Paramètres :
        - img_size (tuple) : Taille de l'image (hauteur, largeur).
        
        Retourne :
        - out_size (tuple) : Taille de sortie (hauteur, largeur).
        """
        ar = img_size[0] / img_size[1]

        min_prod = 100
        max_prod = 120
        ar_array = []
        size_array = []
        for n1 in range(7, 14):
            for n2 in range(7, 14):
                if min_prod <= n1 * n2 <= max_prod:
                    this_ar = n1 / n2
                    this_ar_ratio = min((ar, this_ar)) / max((ar, this_ar))
                    ar_array.append(this_ar_ratio)
                    size_array.append((n1, n2))

        max_ar_ratio_idx = np.argmax(np.array(ar_array)).item()
        bn_size = size_array[max_ar_ratio_idx]
        out_size = tuple(r * 32 for r in bn_size)
        return (416, 288)
        return out_size

    def get_out_size_train(self, img_size):
        """
        Calcule la taille de sortie pour l'entraînement en fonction de la taille de l'image.

        Paramètres :
        - img_size (tuple) : Taille de l'image (hauteur, largeur).
        
        Retourne :
        - out_size (tuple) : Taille de sortie (hauteur, largeur).
        """
        selection = (8, 13), (9, 13), (9, 12), (12, 9), (13, 9)
        ar = img_size[0] / img_size[1]
        ar_array = []
        size_array = []
        for n1, n2 in selection:
            this_ar = n1 / n2
            this_ar_ratio = min((ar, this_ar)) / max((ar, this_ar))
            ar_array.append(this_ar_ratio)
            size_array.append((n1, n2))

        max_ar_ratio_idx = np.argmax(np.array(ar_array)).item()
        bn_size = size_array[max_ar_ratio_idx]
        out_size = tuple(r * 32 for r in bn_size)
        return (416, 288)

        return out_size

    def load_data(self): 
        """
        Charge les données d'image et leurs tailles.

        Retourne :
        - all_image_files (list) : Liste des fichiers d'image et de leurs cartes associées.
        - size_dict (dict) : Dictionnaire des tailles d'image.
        """
        all_image_files = []
        print("test image dir: ",self.img_dir)
        for img_file in sorted(self.img_dir.glob("*.jpg")):

            all_image_files.append({
                'img': img_file.name,
                'map': img_file.stem + "_saillance.jpg",  #A modifier en fonction de ce qui est mis dans la base de données
                'pts': img_file.stem + "_fixPts.jpg",#A modifier en fonction de ce qui est mis dans la base de données
            })


            #print("l'image:",img_file.name)
            assert((self.sal_dir / all_image_files[-1]['map']).exists())
            assert((self.fix_dir / all_image_files[-1]['pts']).exists()) #Pb ici car j'ai pas de cartes de fixation

        size_dict = {}
        for img_idx in range(self.n_train_val_images):
            img = cv2.imread(
                str(self.img_dir / all_image_files[img_idx]['img']))
            size_dict[img_idx] = {'img_size': img.shape[:2]}

        for img_idx in self.samples:
            img_size = size_dict[img_idx]['img_size']
            if self.phase in ('train', 'valid'):
                out_size = self.get_out_size_train(img_size)
            else:
                out_size = self.get_out_size_eval(img_size)


            if self.phase in ('train', 'valid'):
                target_size = tuple(sz * 2 for sz in out_size)
            else:
                target_size = img_size

            size_dict[img_idx].update({
                'out_size': out_size, 'target_size': target_size})

        return all_image_files, size_dict

    def __len__(self):
        return len(self.samples)

    def preprocess(self, img, out_size=None, data='img'):
        transformations = [
            transforms.ToPILImage(),
        ]
        if data in ('img', 'sal'):
            transformations.append(transforms.Resize(
                out_size, interpolation=PIL.Image.LANCZOS))
        else:
            transformations.append(transforms.Resize(
                out_size, interpolation=PIL.Image.NEAREST))
        transformations.append(transforms.ToTensor())
        if data == 'img' and 'rgb_mean' in self.preproc_cfg:
            transformations.append(
                transforms.Normalize(
                    self.preproc_cfg['rgb_mean'], self.preproc_cfg['rgb_std']))
        elif data == 'sal':
            transformations.append(transforms.Lambda(normalize_tensor))
        elif data == 'fix':
            transformations.append(
                transforms.Lambda(lambda fix: torch.gt(fix, 0.1)))

        processing = transforms.Compose(transformations)
        tensor = processing(img)
        return tensor

    def get_data(self, img_idx):
        img = self.get_img(img_idx)
        out_size = self.size_dict[img_idx]['out_size']
        target_size = self.target_size_dict[img_idx]
        img = self.preprocess(img, out_size=out_size, data='img')
        if self.phase == 'test':
            return img, target_size

        sal = self.get_map(img_idx)
        sal = self.preprocess(sal, target_size, data='sal')
        fix = self.get_fixation_map(img_idx) 
        fix = self.preprocess(fix, target_size, data='fix')
        return img, sal, fix, target_size

    def __getitem__(self, item):
        img_idx = self.samples[item]
        return self.get_data(img_idx)

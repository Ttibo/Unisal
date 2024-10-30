from pathlib import Path
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
    return tensor / tsum if tsum > 0 else tensor.fill_(1. / tensor.numel())

class ImageDataset(Dataset):
    n_train_val_images = 64
    dynamic = False

    def __init__(self, path, subset=None, verbose=1, preproc_cfg=None):
        self.subset = subset
        self.verbose = verbose
        self.preproc_cfg = {
            'rgb_mean': (0.485, 0.456, 0.406),
            'rgb_std': (0.229, 0.224, 0.225),
        }
        if preproc_cfg is not None:
            self.preproc_cfg.update(preproc_cfg)
        self.dir = Path(path)
        self.all_image_files, self.size_dict = self.load_data()
        self.img_size = (288,416)
        self.target_size = (360,520)

    def get_map(self, img_idx):
        map_file = self.sal_dir / self.all_image_files[img_idx]['map']
        return cv2.imread(str(map_file), cv2.IMREAD_GRAYSCALE)

    def get_img(self, img_idx):
        img_file = self.img_dir / self.all_image_files[img_idx]['img']
        return np.ascontiguousarray(cv2.imread(str(img_file))[:, :, ::-1])

    def get_fixation_map(self, img_idx):
        fix_map_file = self.fix_dir / self.all_image_files[img_idx]['pts']
        return cv2.imread(str(fix_map_file), cv2.IMREAD_GRAYSCALE)

    @property
    def fix_dir(self): return self.dir / 'fixations'
    @property
    def sal_dir(self): return self.dir / 'maps'
    @property
    def img_dir(self): return self.dir / 'images'

    def get_out_size(self, img_size, train=True):
        if train:
            selection = [(8, 13), (9, 13), (9, 12), (12, 9), (13, 9)]
        else:
            selection = [(n1, n2) for n1 in range(7, 14) for n2 in range(7, 14) if 100 <= n1 * n2 <= 120]
        
        ar = img_size[0] / img_size[1]
        best_size = max(selection, key=lambda s: min(ar, s[0] / s[1]) / max(ar, s[0] / s[1]))

        return (288,416)
        return tuple(r * 32 for r in best_size)

    def load_data(self):
        all_image_files = []
        for img_file in sorted(self.img_dir.glob("*.jpg")):
            all_image_files.append({
                'img': img_file.name,
                'map': img_file.stem + "_saillance.jpg",
                'pts': img_file.stem + "_fixPts.jpg",
            })

        # size_dict = {i: {'img_size': cv2.imread(str(self.img_dir / f['img'])).shape[:2]} for i, f in enumerate(all_image_files)}
        size_dict = {i: {'img_size': (416, 600)} for i, f in enumerate(all_image_files)}
        return all_image_files, size_dict

    def __len__(self):
        return len(self.all_image_files)

    def preprocess(self, img, out_size=None, data='img'):
        transformations = [
            transforms.ToPILImage(),
            transforms.Resize(out_size, interpolation=PIL.Image.LANCZOS if data != 'fix' else PIL.Image.NEAREST), transforms.ToTensor()
            ]
        if data == 'img':
            transformations.append(transforms.Normalize(
                self.preproc_cfg['rgb_mean'],
                self.preproc_cfg['rgb_std']
                ))
        elif data == 'sal':
            transformations.append(transforms.Lambda(normalize_tensor))
        elif data == 'fix':
            transformations.append(transforms.Lambda(lambda fix: torch.gt(fix, 0.1)))
        return transforms.Compose(transformations)(img)

    def get_data(self, img_idx):
        img = self.preprocess(self.get_img(img_idx), out_size=self.img_size)
        sal = self.preprocess(self.get_map(img_idx), out_size=self.target_size, data='sal')
        fix = self.preprocess(self.get_fixation_map(img_idx), out_size=self.target_size, data='fix')

        return img, sal, fix, self.target_size

    def __getitem__(self, idx):
        return self.get_data(idx)

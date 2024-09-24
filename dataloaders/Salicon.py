from pathlib import Path
import os
import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler, SequentialSampler
from torchvision import transforms
import numpy as np
import cv2
import PIL
import scipy.io


preproc_cfg = {
    'rgb_mean': (0.485, 0.456, 0.406),
    'rgb_std': (0.229, 0.224, 0.225),
}

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

class SALICONDataset(Dataset):

    def __init__(self, path , phase='train', subset=None, verbose=1,
                 out_size=(288, 384), target_size=(480, 640),
                 preproc_cfg=None):
        self.phase = phase
        self.train = phase == 'train'
        self.subset = subset
        self.verbose = verbose
        self.out_size = out_size
        self.target_size = target_size
        self.dir = Path(path) # path dataset
        self.preproc_cfg = {
            'rgb_mean': (0.485, 0.456, 0.406),
            'rgb_std': (0.229, 0.224, 0.225),
        }

        if preproc_cfg is not None:
            self.preproc_cfg.update(preproc_cfg)
        self.phase_str = 'val' if phase in ('valid', 'eval') else phase
        self.file_stem = f"COCO_{self.phase_str}2014_"
        self.file_nr = "{:012d}"

        self.samples = self.prepare_samples()
        if self.subset is not None:
            self.samples = self.samples[:int(len(self.samples) * subset)]
        # For compatibility with video datasets
        self.n_images_dict = {img_nr: 1 for img_nr in self.samples}
        self.target_size_dict = {
            img_nr: self.target_size for img_nr in self.samples}
        self.n_samples = len(self.samples)
        self.frame_modulo = 1

    def get_map(self, img_nr):
        map_file = self.dir / 'maps' / self.phase_str / (
                self.file_stem + self.file_nr.format(img_nr) + '.png')
        map = cv2.imread(str(map_file), cv2.IMREAD_GRAYSCALE)
        assert(map is not None)
        return map

    def get_img(self, img_nr):
        img_file = self.dir / 'images' / self.phase_str / (
                self.file_stem + self.file_nr.format(img_nr) + '.jpg')
        img = cv2.imread(str(img_file))
        assert(img is not None)
        return np.ascontiguousarray(img[:, :, ::-1])

    def get_raw_fixations(self, img_nr):
        raw_fix_file = self.dir / 'fixations' / self.phase_str / (
                self.file_stem + self.file_nr.format(img_nr) + '.mat')
        fix_data = scipy.io.loadmat(raw_fix_file)
        fixations_array = [gaze[2] for gaze in fix_data['gaze'][:, 0]]
        return fixations_array, fix_data['resolution'].tolist()[0]

    def process_raw_fixations(self, fixations_array, res):
        fix_map = np.zeros(res, dtype=np.uint8)
        for subject_fixations in fixations_array:
            fix_map[subject_fixations[:, 1] - 1, subject_fixations[:, 0] - 1]\
                = 255
        return fix_map

    def get_fixation_map(self, img_nr):
        fix_map_file = self.dir / 'fixations' / self.phase_str / (
                self.file_stem + self.file_nr.format(img_nr) + '.png')
        if fix_map_file.exists():
            fix_map = cv2.imread(str(fix_map_file), cv2.IMREAD_GRAYSCALE)
        else:
            fixations_array, res = self.get_raw_fixations(img_nr)
            fix_map = self.process_raw_fixations(fixations_array, res)
            cv2.imwrite(str(fix_map_file), fix_map)
        return fix_map

    def prepare_samples(self):
        samples = []
        for index , file in enumerate((self.dir / 'images' / self.phase_str).glob(self.file_stem + '*.jpg')):
            samples.append(int(file.stem[-12:]))
            if index == 100 : 
                break

        print("Samples Salicon " , len(samples))
        return sorted(samples)

    def __len__(self):
        return len(self.samples)

    def preprocess(self, img, data='img'):
        transformations = [
            transforms.ToPILImage(),
        ]
        if data == 'img':
            transformations.append(transforms.Resize(
                self.out_size, interpolation=PIL.Image.LANCZOS))
        transformations.append(transforms.ToTensor())
        if data == 'img' and 'rgb_mean' in self.preproc_cfg:
            transformations.append(
                transforms.Normalize(
                    self.preproc_cfg['rgb_mean'], self.preproc_cfg['rgb_std']))
        elif data == 'sal':
            transformations.append(transforms.Lambda(normalize_tensor))
        elif data == 'fix':
            transformations.append(
                transforms.Lambda(lambda fix: torch.gt(fix, 0.5)))

        processing = transforms.Compose(transformations)
        tensor = processing(img)
        return tensor

    def get_data(self, img_nr):
        img = self.get_img(img_nr)
        img = self.preprocess(img, data='img')
        if self.phase == 'test':
            return [1], img, self.target_size

        sal = self.get_map(img_nr)
        sal = self.preprocess(sal, data='sal')
        fix = self.get_fixation_map(img_nr)
        fix = self.preprocess(fix, data='fix')

        return img, sal, fix, self.target_size

    def __getitem__(self, item):
        img_nr = self.samples[item]
        return self.get_data(img_nr)


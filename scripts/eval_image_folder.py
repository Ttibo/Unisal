import sys
sys.path.append("./")
sys.path.append("../")


import utils

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import torch
from torchvision import transforms
import numpy as np
import cv2
import PIL
import random
import os
import re
import glob

import json
import model
import dataloaders

if torch.cuda.is_available():
    DEFAULT_DEVICE = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    DEFAULT_DEVICE = torch.device("mps")
else:
    DEFAULT_DEVICE = torch.device("cpu")

def normalize_tensor(tensor, rescale=False):
    tmin = torch.min(tensor)
    if rescale or tmin < 0:
        tensor -= tmin
    tsum = tensor.sum()
    return tensor / tsum if tsum > 0 else tensor.fill_(1. / tensor.numel())

class Saliency:
    def __init__(self, pathModel : str = "/weights/packging_3s/"):
        assert( os.path.exists(pathModel)) , " Error folder model weights"

        with open(pathModel + "sources.json", 'r') as file:
            sources = json.load(file) 

        self.path_ = os.path.dirname(os.path.abspath(__file__))
        self.model = model.UNISAL(bypass_rnn=False, sources=sources)

        self.model.load_weights(pathModel + "weights_best.pth")
        self.model.to(DEFAULT_DEVICE)

        self.preproc_cfg = {
            'rgb_mean': (0.485, 0.456, 0.406),
            'rgb_std': (0.229, 0.224, 0.225),
        }

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

    def normalize(self , image : np.ndarray ) -> np.ndarray: 
        # normalize image btw 0 and 1 : -> float32
        print("normalize : " ,image.shape)
        return cv2.normalize(image, image, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    
    def show_maps(self , img , colormap ,map):
        # Créer une figure avec 4 sous-graphiques
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        # Afficher la première image
        axs[0].imshow(img)
        axs[0].set_title('Image 1')
        axs[0].axis('off')  # Pour ne pas afficher les axes

        # Afficher la deuxième image (colormap_)
        axs[1].imshow(colormap)
        axs[1].set_title('Colormap')
        axs[1].axis('off')

        # Afficher la troisième image (map_)
        axs[2].imshow(map)
        axs[2].set_title('Map')
        axs[2].axis('off')

    def run(self , pathImage : str, source : str ) -> np.ndarray:

        img = cv2.imread(str(pathImage))
        img_tensor = self.preprocess(img, out_size=(288, 384), data='img')
        map_ = self.image_inference(img_tensor, source)

        # smap = np.exp(map_)
        # smap = np.squeeze(smap)
        # smap = smap
        # map_ = (smap / np.amax(smap) )
        # map_ = cv2.resize(map_ , (img.shape[1] , img.shape[0]))
        
        # predicted_colored = cv2.applyColorMap(map_.astype(np.uint8) , cv2.COLORMAP_JET)
        # res_ = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0.3, cv2.cvtColor(predicted_colored, cv2.COLOR_BGR2RGB), 0.7, 0.0)

        return map_

        # self.show_maps(img , map_ , res_)

    def image_inference(self , img : torch.Tensor, source : str ) -> np.ndarray :
        img_ = img.to(DEFAULT_DEVICE).unsqueeze(0).unsqueeze(0)
        map_ = self.model(img_, source = source)
        return map_

def eval_sequences(
        pred_seq,
        sal_seq,
        fix_seq,
        metrics = ['sim' , 'aucj'],
        other_maps=None,
        auc_portion=1.0
    ):

    """
    Compute SIM, AUC-J and s-AUC scores
    """

    # process inputs
    metrics = [metric for metric in metrics if metric in ("sim", "aucj", "aucs")]
    if "aucs" in metrics:
        assert other_maps is not None

    # Preprocess sequences
    shape = pred_seq.shape
    new_shape = (-1, shape[-2], shape[-1])
    pred_seq = pred_seq.exp()
    pred_seq = pred_seq.detach().cpu().numpy().reshape(new_shape)
    sal_seq = sal_seq.detach().cpu().numpy().reshape(new_shape)
    fix_seq = fix_seq.detach().cpu().numpy().reshape(new_shape)

    auc_indices = set(list(range(shape[1])))

    # Compute the metrics
    results = {metric: [] for metric in metrics}
    for idx, (pred, sal, fix) in enumerate(zip(pred_seq, sal_seq, fix_seq)):
        for this_metric in metrics:
            if this_metric == "sim":
                results["sim"].append(similarity(pred, sal))
            if this_metric == "aucj":
                if idx in auc_indices:
                    results["aucj"].append(auc_judd(pred, fix))
            if this_metric == "aucs":
                if idx in auc_indices:
                    other_map = next(other_maps)
                    results["aucs"].append(
                        auc_shuff_acl(pred, fix, other_map)
                    )
    return [np.array(results[metric]) for metric in metrics]



from pathlib import Path
def extract_images_from_folder(path : str):
    dir_img = Path(path + "images")
    dir_sal = Path(path + "maps")
    dir_fix = Path(path + "fixations")
    

    # extract all images form images
    all_image_files = []
    for img_file in sorted(dir_img.glob("*.jpg")):
        all_image_files.append({
            'img': os.path.join(dir_img , img_file.name),
            'map': os.path.join(dir_sal , img_file.stem + "_saillance.jpg"),
            'pts': os.path.join(dir_fix , img_file.stem + "_fixPts.jpg")
        })

    return all_image_files 


def normalize_map(s_map):
    # normalize the salience map (as done in MIT code)
    norm_s_map = (s_map - np.min(s_map)) / ((np.max(s_map) - np.min(s_map)))
    return norm_s_map


def auc_judd(s_map, gt):
    # ground truth is discrete, s_map is continous and normalized
    s_map = normalize_map(s_map)

    assert np.max(gt) == 1.0,\
        'Ground truth not discretized properly max value > 1.0'
    assert np.max(s_map) == 1.0,\
        'Salience map not normalized properly max value > 1.0'

    # thresholds are calculated from the salience map,
    # only at places where fixations are present
    thresholds = s_map[gt > 0].tolist()

    num_fixations = len(thresholds)
    # num fixations is no. of salience map values at gt >0

    thresholds = sorted(set(thresholds))

    area = []
    area.append((0.0, 0.0))
    for thresh in thresholds:
        # in the salience map,
        # keep only those pixels with values above threshold
        temp = s_map >= thresh
        num_overlap = np.sum(np.logical_and(temp, gt))
        tp = num_overlap / (num_fixations * 1.0)

        # total number of pixels > threshold - number of pixels that overlap
        # with gt / total number of non fixated pixels
        # this becomes nan when gt is full of fixations..this won't happen
        fp = (np.sum(temp) - num_overlap) / (np.prod(gt.shape[:2]) - num_fixations)

        area.append((round(tp, 4) ,round(fp, 4)))

    area.append((1.0, 1.0))
    area.sort(key=lambda x: x[0])
    tp_list, fp_list = list(zip(*area))
    return np.trapz(np.array(tp_list), np.array(fp_list))


def auc_shuff_acl(s_map, gt, other_map, n_splits=100, stepsize=0.1):

    # If there are no fixations to predict, return NaN
    if np.sum(gt) == 0:
        print('no gt')
        return None

    # normalize saliency map
    s_map = normalize_map(s_map)

    S = s_map.flatten()
    F = gt.flatten()
    Oth = other_map.flatten()

    Sth = S[F > 0]  # sal map values at fixation locations
    Nfixations = len(Sth)

    # for each fixation, sample Nsplits values from the sal map at locations
    # specified by other_map

    ind = np.where(Oth > 0)[0]  # find fixation locations on other images

    Nfixations_oth = min(Nfixations, len(ind))
    randfix = np.full((Nfixations_oth, n_splits), np.nan)

    for i in range(n_splits):
        # randomize choice of fixation locations
        randind = np.random.permutation(ind.copy())
        # sal map values at random fixation locations of other random images
        randfix[:, i] = S[randind[:Nfixations_oth]]

    # calculate AUC per random split (set of random locations)
    auc = np.full(n_splits, np.nan)
    for s in range(n_splits):

        curfix = randfix[:, s]

        allthreshes = np.flip(np.arange(0, max(np.max(Sth), np.max(curfix)), stepsize))
        tp = np.zeros(len(allthreshes) + 2)
        fp = np.zeros(len(allthreshes) + 2)
        tp[-1] = 1
        fp[-1] = 1

        for i in range(len(allthreshes)):
            thresh = allthreshes[i]
            tp[i + 1] = np.sum(Sth >= thresh) / Nfixations
            fp[i + 1] = np.sum(curfix >= thresh) / Nfixations_oth

        auc[s] = np.trapz(np.array(tp), np.array(fp))

    return np.mean(auc)


def similarity(s_map, gt):
    return np.sum(np.minimum(s_map, gt))


if __name__ == "__main__":
    file_ = "/Users/coconut/Documents/Dataset/GenSaliency/test/image_1.jpg"

    saliency_ = Saliency( pathModel = "../weights/fine_tune_ittention_v1/")


    all_images_files = extract_images_from_folder("/Users/coconut/Documents/Dataset/GenSaliency/VisualSaliency/Packaging_delta_3_sigma_20/train/") 

    for img in all_images_files:
        pred_map_ = saliency_.run(img['img'] , source="SALICON")
        map_ = cv2.imread(img['map'], 0)
        fix_ = cv2.imread(img['pts'], 0)

        sal_tensor = saliency_.preprocess(map_, out_size=(288, 384), data='sal').unsqueeze(0).unsqueeze(0)
        fix_tensor = saliency_.preprocess(fix_, out_size=(288, 384), data='fix').unsqueeze(0).unsqueeze(0)

        print(pred_map_.shape)
        print(sal_tensor.shape)
        print(fix_tensor.shape)

        # map_ = (map_.astype(np.float32) / np.amax(map_))
        # fix_ = (fix_.astype(np.float32) / np.amax(fix_))
        # pred_map_ = (pred_map_.astype(np.float32) / np.amax(pred_map_))

        eval = eval_sequences(
            pred_seq = pred_map_,
            sal_seq = sal_tensor , 
            fix_seq = fix_tensor
        )

        print(eval)

        cv2.imshow("pred map" , pred_map_)
        cv2.imshow("map" , map_)
        cv2.imshow("fix" , fix_)
        cv2.waitKey()

        break


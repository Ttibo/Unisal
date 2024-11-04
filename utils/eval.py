import sys
import torch 
import argparse
sys.path.append("./")
sys.path.append("../")




from torch.utils.data import Dataset, DataLoader
from model import UNISAL
from pathlib import Path


if torch.cuda.is_available():
    DEFAULT_DEVICE = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    DEFAULT_DEVICE = torch.device("mps")
else:
    DEFAULT_DEVICE = torch.device("cpu")

import random
import numpy as np

def normalize_tensor(tensor, rescale=False):
    tmin = torch.min(tensor)
    if rescale or tmin < 0:
        tensor -= tmin
    tsum = tensor.sum()
    return tensor / tsum if tsum > 0 else tensor.fill_(1. / tensor.numel())


def eval_sequences(
    pred_seq, sal_seq, fix_seq, metrics =  ("sim", "aucj"), other_maps=None, auc_portion=1.0
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

    # Optionally compute AUC-s for a subset of frames to reduce runtime
    if auc_portion < 1:
        auc_indices = set(
            random.sample(range(shape[1]), max(1, int(auc_portion * shape[1])))
        )
    else:
        auc_indices = set(list(range(shape[1])))

    # Compute the metrics
    results = {metric: [] for metric in metrics}
    for idx, (pred, sal, fix) in enumerate(zip(pred_seq, sal_seq, fix_seq)):
        for this_metric in metrics:
            if this_metric == "sim":
                results["sim"].append(float(similarity(pred, sal)))
            if this_metric == "aucj":
                if idx in auc_indices:
                    results["aucj"].append(float(auc_judd(pred, fix)))
            if this_metric == "aucs":
                if idx in auc_indices:
                    other_map = next(other_maps)
                    results["aucs"].append(
                        auc_shuff_acl(pred, fix, other_map)
                    )

    print(results)
    return results 


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




def load_data(path):

    img_dir = path + "images/"
    map_dir = path + "maps/"
    fix_dir = path + "fixations/"

    all_image_files = []
    for img_file in sorted(Path(img_dir).glob("*.jpg")):
        all_image_files.append({
            'img': img_dir + img_file.name,
            'map': map_dir + img_file.stem + "_saillance.jpg",
            'pts': fix_dir + img_file.stem + "_fixPts.jpg",
        })

    # size_dict = {i: {'img_size': cv2.imread(str(self.img_dir / f['img'])).shape[:2]} for i, f in enumerate(all_image_files)}
    # size_dict = {i: {'img_size': (416, 600)} for i, f in enumerate(all_image_files)}
    return all_image_files


import os 
import json
import cv2
from torchvision import transforms
import PIL


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trainer for the model.')
    parser.add_argument('--path_save', type=str, default="./weights/fine_tune_3sec_ittention_v2/" , help='path save output')

    print("Eval")

    path_ = os.path.dirname(os.path.abspath(__file__))
    path_dataset ="/Users/coconut/Documents/Dataset/GenSaliency/VisualSaliency/Packaging_delta_1_sigma_20/val/" 
    # dataset = dataloaders.ImageDataset(path =path_dataset)

    pathModel = "../weights/fine_tune_1sec_ittention_v2/"
    assert( os.path.exists(pathModel)) , " Error folder model weights"

    with open(pathModel + "sources.json", 'r') as file:
        sources = json.load(file) 

    model = UNISAL(bypass_rnn=False, sources=sources)

    model.load_weights(pathModel + "weights_best.pth")
    model.to(DEFAULT_DEVICE)



    datas_ = load_data(path_dataset)

    preproc_cfg = {
        'rgb_mean': (0.485, 0.456, 0.406),
        'rgb_std': (0.229, 0.224, 0.225),
    }

    img_size = (288,416)
    target_size = (360,520)

    results = {
        "datasets" :[path_dataset],
        "eval" : []

    }
    transformations = [
        transforms.ToPILImage(),
        transforms.Resize(img_size, interpolation=PIL.Image.LANCZOS), transforms.ToTensor()
        ]
    transformations.append(transforms.Normalize(
        preproc_cfg['rgb_mean'],
        preproc_cfg['rgb_std']
        ))

    transformations_fix = [
        transforms.ToPILImage(),
        transforms.Resize(target_size, interpolation=PIL.Image.NEAREST), transforms.ToTensor()
        ]

    transformations_sal = [
        transforms.ToPILImage(),
        transforms.Resize(target_size, interpolation= PIL.Image.LANCZOS), transforms.ToTensor()
        ]

    transformations_sal.append(transforms.Lambda(normalize_tensor))


    transformations_fix.append(transforms.Lambda(lambda fix: torch.gt(fix, 0.1)))




    global_ = {"sim": [],  "aucj" : []}

    for data in datas_:
        print(data)

        img = cv2.imread(str(data['img']))
        map = cv2.imread(data['map'] , cv2.IMREAD_GRAYSCALE)
        fix = cv2.imread(data['pts'] , cv2.IMREAD_GRAYSCALE)

        # preprocess img
        tensor_img = transforms.Compose(transformations)(img)
        tensor_fix = transforms.Compose(transformations_fix)(fix)
        tensor_map = transforms.Compose(transformations_sal)(map)

        tensor_img = tensor_img.to(DEFAULT_DEVICE).unsqueeze(0).unsqueeze(0)
        tensor_fix = tensor_fix.to(DEFAULT_DEVICE).unsqueeze(0).unsqueeze(0)
        tensor_map = tensor_map.to(DEFAULT_DEVICE).unsqueeze(0).unsqueeze(0)



        predict_ = model(
            x = tensor_img,
            target_size = target_size,
            source = "SALICON"
        )


        results_ = eval_sequences(
            pred_seq=predict_,
            sal_seq=tensor_map,
            fix_seq=tensor_fix,
            # metrcis=["sim", "aucj"]
        )

        rr_ = {}
        for key , v in results_.items():
            rr_[key] = float(np.mean(np.asarray(v)))
            global_[key].append(float(np.mean(np.asarray(v))))
        results['eval'].append(rr_)

        # break
    
    
    print("Global rsults")
    for key , v in global_.items():
        global_[key] = float(np.mean(np.asarray(v)))

    print(global_)

    results['global'] = global_

    with open(pathModel + "eval.json", 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)










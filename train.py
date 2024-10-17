

import os
import math
import cv2
import matplotlib.pyplot as plt
import model
import dataloaders
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from itertools import chain
from tqdm import tqdm
from utils.trainer import Trainer


import utils
import torch
import argparse

if torch.cuda.is_available():
    DEFAULT_DEVICE = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    DEFAULT_DEVICE = torch.device("mps")
else:
    DEFAULT_DEVICE = torch.device("cpu")

if __name__ == "__main__":
    # Créez un parser d'arguments
    parser = argparse.ArgumentParser(description='Trainer for the model.')

    # Ajoutez les arguments pour le Trainer
    parser.add_argument('--num_epochs', type=int, default=2, help='Nombre d\'époques pour l\'entraînement.')
    parser.add_argument('--optim_algo', type=str, default="SGD", help='Algorithme d\'optimisation.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum pour l\'optimiseur.')
    parser.add_argument('--lr', type=float, default=0.04, help='Taux d\'apprentissage.')
    parser.add_argument('--lr_scheduler', type=str, default="ExponentialLR", help='Type de planificateur de taux d\'apprentissage.')
    parser.add_argument('--lr_gamma', type=float, default=0.99, help='Facteur gamma pour le planificateur de taux d\'apprentissage.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Poids de décroissance pour l\'optimiseur.')
    parser.add_argument('--cnn_weight_decay', type=float, default=1e-5, help='Poids de décroissance pour le CNN.')
    parser.add_argument('--train_cnn_after', type=int, default=100, help='Nombres epochs pour commencer à entrainer l encoder')
    parser.add_argument('--grad_clip', type=float, default=2.0, help='Valeur de coupure de gradient.')
    parser.add_argument('--cnn_lr_factor', type=float, default=0.1, help='Facteur de taux d\'apprentissage pour le CNN.')
    parser.add_argument('--loss_metrics', type=str, nargs='+', default=["kld", "nss", "cc"], help='Métriques de perte à utiliser.')
    parser.add_argument('--loss_weights', type=float, nargs='+', default=[1, -0.1, -0.1], help='Poids des métriques de perte.')
    parser.add_argument('--chkpnt_warmup', type=int, default=2, help='Époques de montée en température pour le point de contrôle.')
    parser.add_argument('--chkpnt_epochs', type=int, default=2, help='Nombre d\'époques pour sauvegarder le point de contrôle.')
    parser.add_argument('--path_save', type=str, default="./weights/video_test/" , help='path save output')
    parser.add_argument('--path_dataset_image', type=str, default=None , help='path dataset')
    parser.add_argument('--path_dataset_video', type=str, default=None , help='path dataset')

    # Analysez les arguments
    args = parser.parse_args()
    print(args)
    # create model Unisal
    unisal_ = model.UNISAL(bypass_rnn=False)

    # load model from github and res
    directory_ = "./weights/weights_best.pth"
    unisal_.load_weights(directory_ )
    # move model to device
    print(f"Move model to torch device set to: {DEFAULT_DEVICE}")
    unisal_.to(DEFAULT_DEVICE)

    dataloaders_ = []

    if args.path_dataset_image is not None:
        packaging_train = dataloaders.PACKAGINGDataset(path=args.path_dataset_image + "/train/")
        packaging_val = dataloaders.PACKAGINGDataset(path=args.path_dataset_image + "/val/")

        dataloaders_.append(
        {
            'name' : 'Packaging',
            'loader' : {
                'train' : DataLoader(packaging_train, batch_size=10, shuffle=True),
                'val' : DataLoader(packaging_val, batch_size=10, shuffle=True)
            }
        })

    if args.path_dataset_video is not None:
        video_train = dataloaders.VideoDataset(path=args.path_dataset_video  + "/train/", N = 12)
        video_val = dataloaders.VideoDataset(path=args.path_dataset_video + "/val/", N=12)
        dataloaders_.append(
            {
                'name' : 'Video',
                'loader' : {
                    'train' : DataLoader(video_train, batch_size=4, shuffle=True),
                    'val' : DataLoader(video_val, batch_size=4, shuffle=True)
                }
            })

    assert(len(dataloaders_) != 0) , "Error no data found"

    # Instanciez le Trainer avec les arguments
    trainer = Trainer(
        dataloaders=dataloaders_,  # Remplacez ceci par vos dataloaders
        device=DEFAULT_DEVICE,  # Ou tout autre dispositif
        model=unisal_,  # Remplacez ceci par votre modèle
        path=args.path_save,  # Remplacez ceci par le chemin vers les points de contrôle
        num_epochs=args.num_epochs,
        optim_algo=args.optim_algo,
        momentum=args.momentum,
        lr=args.lr,
        lr_scheduler=args.lr_scheduler,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
        cnn_weight_decay=args.cnn_weight_decay,
        grad_clip=args.grad_clip,
        cnn_lr_factor=args.cnn_lr_factor,
        loss_metrics=args.loss_metrics,
        loss_weights=args.loss_weights,
        chkpnt_warmup=args.chkpnt_warmup,
        chkpnt_epochs=args.chkpnt_epochs,
        train_cnn_after=args.train_cnn_after
    )

    trainer.fit()

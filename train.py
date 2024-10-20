

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
    parser.add_argument('--num_epochs', type=int, default=25, help='Nombre d\'époques pour l\'entraînement.')
    parser.add_argument('--batch_size_image', type=int, default=10, help='Batch size image')
    parser.add_argument('--batch_size_video', type=int, default=4, help='Batch size video')
    parser.add_argument('--seq_len', type=int, default=12, help='sequence lenght video')
    parser.add_argument('--pretrained', type=bool, default=False, help='load pretrained model')
    parser.add_argument('--optim_algo', type=str, default="SGD", help='Algorithme d\'optimisation.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum pour l\'optimiseur.')
    parser.add_argument('--lr', type=float, default=0.04, help='Taux d\'apprentissage.')
    parser.add_argument('--lr_scheduler', type=str, default="ExponentialLR", help='Type de planificateur de taux d\'apprentissage.')
    parser.add_argument('--lr_gamma', type=float, default=0.99, help='Facteur gamma pour le planificateur de taux d\'apprentissage.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Poids de décroissance pour l\'optimiseur.')
    parser.add_argument('--cnn_weight_decay', type=float, default=1e-5, help='Poids de décroissance pour le CNN.')
    parser.add_argument('--train_cnn_after', type=int, default=15, help='Nombres epochs pour commencer à entrainer l encoder')
    parser.add_argument('--grad_clip', type=float, default=2.0, help='Valeur de coupure de gradient.')
    parser.add_argument('--cnn_lr_factor', type=float, default=0.1, help='Facteur de taux d\'apprentissage pour le CNN.')
    parser.add_argument('--loss_metrics', type=str, nargs='+', default=["kld", "nss", "cc"], help='Métriques de perte à utiliser.')
    parser.add_argument('--loss_weights', type=float, nargs='+', default=[1, -0.1, -0.1], help='Poids des métriques de perte.')
    parser.add_argument('--chkpnt_warmup', type=int, default=2, help='Époques de montée en température pour le point de contrôle.')
    parser.add_argument('--chkpnt_epochs', type=int, default=2, help='Nombre d\'époques pour sauvegarder le point de contrôle.')
    parser.add_argument('--path_save', type=str, default="./weights/test_train_scratch/" , help='path save output')

    # Analysez les arguments
    args = parser.parse_args()
    print(args)
    dataloaders_ = []

    # load SALICON dataset
    print("Salicon Dataset")
    salicon_train = dataloaders.SALICONDataset(path ="/Users/coconut/Documents/Dataset/GenSaliency/VisualSaliency/SALICON/", phase="train" )
    salicon_val = dataloaders.SALICONDataset(path ="/Users/coconut/Documents/Dataset/GenSaliency/VisualSaliency/SALICON/", phase="val" )
    dataloaders_.append(
    {
        'name' : 'Salicon',
        'loader' : {
            'train' : DataLoader(salicon_train, batch_size=args.batch_size_image, shuffle=True),
            'val' : DataLoader(salicon_val, batch_size=args.batch_size_image, shuffle=True)
        }
    })

    # load packaging dataset
    print("Packaging Dataset")
    packaging_train = dataloaders.PACKAGINGDataset("/Users/coconut/Documents/Dataset/GenSaliency/VisualSaliency/Packaging_delta_3_sigma_20/train/")
    packaging_val = dataloaders.PACKAGINGDataset("/Users/coconut/Documents/Dataset/GenSaliency/VisualSaliency/Packaging_delta_3_sigma_20/val/")

    dataloaders_.append(
    {
        'name' : 'Packaging',
        'loader' : {
            'train' : DataLoader(packaging_train, batch_size=args.batch_size_image, shuffle=True),
            'val' : DataLoader(packaging_val, batch_size=args.batch_size_image, shuffle=True)
        }
    })

    # Load DHF1K dataset
    print("DHF1K Dataset")
    dhf1k_train = dataloaders.VideoDataset(
        path="/Users/coconut/Documents/Dataset/GenSaliency/VisualSaliency/DHF1K/DHF1K/",
        phase = "train",
        seq_len=args.seq_len
        )

    dhf1k_val = dataloaders.VideoDataset(
        path="/Users/coconut/Documents/Dataset/GenSaliency/VisualSaliency/DHF1K/DHF1K/",
        phase = "val",
        seq_len=args.seq_len
        )

    dataloaders_.append(
        {
            'name' : 'DHF1K',
            'loader' : {
                'train' : DataLoader(dhf1k_train, batch_size=args.batch_size_video, shuffle=True),
                'val' : DataLoader(dhf1k_val, batch_size=args.batch_size_video, shuffle=True)
            }
        })




    # Load UCF dataset
    print("UCF Dataset")
    ucf_train = dataloaders.VideoDataset(
        path="/Users/coconut/Documents/Dataset/GenSaliency/VisualSaliency/ucf_sports_actions/train/",
        phase = "full",
        seq_len=args.seq_len,
        )

    ucf_val = dataloaders.VideoDataset(
        path="/Users/coconut/Documents/Dataset/GenSaliency/VisualSaliency/ucf_sports_actions/val/",
        phase = "full",
        seq_len=args.seq_len
        )

    dataloaders_.append(
        {
            'name' : 'UCF',
            'loader' : {
                'train' : DataLoader(ucf_train, batch_size=args.batch_size_video, shuffle=True),
                'val' : DataLoader(ucf_val, batch_size=args.batch_size_video, shuffle=True)
            }
        })


    # ittention dataset loader video
    print("Advertising Dataset")
    ittention_vid_train = dataloaders.VideoDataset(
        path="/Users/coconut/Documents/Dataset/GenSaliency/VisualSaliency/Ittention_advertising_video/",
        img_dir="frames",
        phase="train",
        seq_len=args.seq_len
        )

    ittention_vid_val = dataloaders.VideoDataset(
        path="/Users/coconut/Documents/Dataset/GenSaliency/VisualSaliency/Ittention_advertising_video/",
        img_dir="frames",
        phase="val",
        seq_len=args.seq_len
        )

    dataloaders_.append(
        {
            'name' : 'Advertising',
            'loader' : {
                'train' : DataLoader(ittention_vid_train, batch_size=args.batch_size_video, shuffle=True),
                'val' : DataLoader(ittention_vid_val, batch_size=args.batch_size_video, shuffle=True)
            }
        })

    assert(len(dataloaders_) != 0) , "Error no data found"



    # create model Unisal
    if args.pretrained:
        unisal_ = model.UNISAL(
            # sources= [loader['name'] for loader in dataloaders_],
            bypass_rnn=False
            )

        directory_ = "./model/weights/weights_best.pth"
        unisal_.load_weights(directory_ )
    else : 
        unisal_ = model.UNISAL(
            sources= [loader['name'] for loader in dataloaders_],
            bypass_rnn=False
            )

    # move model to device
    print(f"Move model to torch device set to: {DEFAULT_DEVICE}")
    unisal_.to(DEFAULT_DEVICE)



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


import setting

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
    parser.add_argument('--num_epochs', type=int, default=40, help='Nombre d\'époques pour l\'entraînement.')
    parser.add_argument('--batch_size_image', type=int, default=20, help='Batch size image')
    parser.add_argument('--batch_size_video', type=int, default=5, help='Batch size video')
    parser.add_argument('--seq_len', type=int, default=18, help='sequence lenght video')
    parser.add_argument('--pretrained', type=bool, default=True, help='load pretrained model')
    parser.add_argument('--optim_algo', type=str, default="SGD", help='Algorithme d\'optimisation.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum pour l\'optimiseur.')
    parser.add_argument('--lr', type=float, default=0.03, help='Taux d\'apprentissage.')
    parser.add_argument('--lr_scheduler', type=str, default="ExponentialLR", help='Type de planificateur de taux d\'apprentissage.')
    parser.add_argument('--lr_gamma', type=float, default=0.95, help='Facteur gamma pour le planificateur de taux d\'apprentissage.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Poids de décroissance pour l\'optimiseur.')
    parser.add_argument('--cnn_weight_decay', type=float, default=1e-5, help='Poids de décroissance pour le CNN.')
    parser.add_argument('--train_cnn_after', type=int, default=0, help='Nombres epochs pour commencer à entrainer l encoder')
    parser.add_argument('--grad_clip', type=float, default=2.0, help='Valeur de coupure de gradient.')
    parser.add_argument('--cnn_lr_factor', type=float, default=0.1, help='Facteur de taux d\'apprentissage pour le CNN.')
    parser.add_argument('--loss_metrics', type=str, nargs='+', default=["kld", "nss", "cc"], help='Métriques de perte à utiliser.')
    parser.add_argument('--loss_weights', type=float, nargs='+', default=[1, -0.1, -0.1], help='Poids des métriques de perte.')
    parser.add_argument('--chkpnt_warmup', type=int, default=2, help='Époques de montée en température pour le point de contrôle.')
    parser.add_argument('--chkpnt_epochs', type=int, default=2, help='Nombre d\'époques pour sauvegarder le point de contrôle.')
    parser.add_argument('--path_save', type=str, default="./weights/fine_tune_ittention_v1/" , help='path save output')
    parser.add_argument('--setting', type=str, default="local" , help='local or server setting')


    # Analysez les arguments
    args = parser.parse_args()
    print(args)

    if args.setting == "local":
        path_dataset_ = setting.DATASET_PATHS_LOCAL
    if args.setting == "server":
        path_dataset_ = setting.DATASET_PATHS_SERVER
    if args.setting == "desktop":
        path_dataset_ = setting.DATASET_PATHS_DESKTOP
    print(path_dataset_)


    dataloaders_ = {}

    for key, v in path_dataset_.items():
        print(key , " " , v)

        # if v['type'] == "image" and key == "SALICON":
        #     _train = dataloaders.SALICONDataset(path =v['train'], phase="train" )
        #     _val = dataloaders.SALICONDataset(path =v['val'], phase="val" )

        # elif v['type'] == "image":
        if v['type'] == "image":
            _train = dataloaders.ImageDataset(path =v['train'] )
            _val = dataloaders.ImageDataset(path =v['val'])

        elif v['type'] == "video" and key == "UCFSports":
            _train = dataloaders.VideoDataset(path = v['path'] + "train/" , seq_len= args.seq_len, frame_modulo= 3,ratio_val_test = 1., phase= "train" , extension=v['extension'], img_dir = v['img_dir'])
            _val = dataloaders.VideoDataset(path = v['path'] + "val/" , seq_len= args.seq_len, frame_modulo= 3,ratio_val_test=0., phase= "val" , extension=v['extension'], img_dir = v['img_dir'])

        elif v['type'] == "video":
            _train = dataloaders.VideoDataset(path = v['path'], seq_len= args.seq_len, frame_modulo= 3, phase= "train" , extension=v['extension'], img_dir = v['img_dir'])
            _val = dataloaders.VideoDataset(path = v['path'] , seq_len= args.seq_len, frame_modulo= 3, phase= "val" , extension=v['extension'], img_dir = v['img_dir'])

        batch_size = args.batch_size_video if v['type'] == "video" else args.batch_size_image
        print(f" - Batch size {batch_size}")
        print(f" - len train {len(_train)} : val  {len(_val)})")

        dataloaders_[key] = {
            'train' : DataLoader(_train, batch_size=batch_size, shuffle=True),
            'val' : DataLoader(_val, batch_size=batch_size, shuffle=True)
        }

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
            sources= [key for key , _ in dataloaders_.items()],
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

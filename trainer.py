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


import utils
import torch

if torch.cuda.is_available():
    DEFAULT_DEVICE = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    DEFAULT_DEVICE = torch.device("mps")
else:
    DEFAULT_DEVICE = torch.device("cpu")

class Trainer():
    def __init__(
        self,
        dataloaders,
        device,
        model ,
        path ,

        num_epochs=100,
        optim_algo="SGD",
        momentum=0.9,
        lr=0.04,
        lr_scheduler="ExponentialLR",
        lr_gamma=0.99,
        weight_decay=1e-4,
        cnn_weight_decay=1e-5,
        grad_clip=2.0,
        cnn_lr_factor=0.1,
        loss_metrics=("kld", "nss", "cc"),
        loss_weights=(1, -0.1, -0.1),
        chkpnt_warmup=2,
        chkpnt_epochs=2,

    ):
        self.dataloaders = dataloaders
        self.device = device
        self.num_epochs = num_epochs
        self.optim_algo = optim_algo
        self.momentum = momentum
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.cnn_weight_decay = cnn_weight_decay
        self.grad_clip = grad_clip
        self.loss_metrics = loss_metrics
        self.loss_weights = loss_weights
        self.cnn_lr_factor = cnn_lr_factor
        self.model = model
        self.chkpnt_warmup = chkpnt_warmup
        self.chkpnt_epochs = chkpnt_epochs

        self.phases = ("train", "val")
        self.train_cnn_after = 0
        self.cnn_eval = True
        self.loss_weights = (1,-0.1 , -0.1)

        self.train_dir = Path(path)
        if os.path.exists(path) == False:
            os.mkdir(self.train_dir)

        if os.path.exists(self.train_dir / 'checkpoints' ) == False:
            os.mkdir(self.train_dir / 'checkpoints')

        self.epoch = 0
        self.phase = None

        self._datasets = {}
        self._dataloaders = {}

        self._scheduler = None
        self._optimizer = None

        self._model = None

        self.best_epoch = 0
        self.best_val_score = None


    def fit(self):
        """
        Train the model
        """

        while self.epoch < self.num_epochs:
            self.scheduler.step(epoch=self.epoch)
            lr = self.optimizer.param_groups[0]["lr"]
            print(f"\nEpoch {self.epoch:3d}, lr {lr:.5f}")

            for self.phase in self.phases:
                self.fit_phase()

            # Save a checkpoint if applicable
            if (
                self.epoch >= self.chkpnt_warmup
                and (self.epoch + 1) % self.chkpnt_epochs == 0
            ) or self.epoch == self.num_epochs - 1:
                self.save_chkpnt()

            self.epoch += 1

    def fit_phase(self):
        """
        Run the current phase (training or validation)
        """

        # Prepare book keeping
        running_losses = {src['name']: 0.0 for src in self.dataloaders}
        running_loss_summands = {
            src['name']: [0.0 for _ in self.loss_weights] for src in self.dataloaders
        }
        n_samples = {src['name']: 0 for src in self.dataloaders}
        # Set model train/eval mode
        self.model.train(self.phase == "train")

        # Switch CNN gradients on/off and set CNN eval mode (for BN modules)
        if self.phase == "train":
            cnn_grad = self.epoch >= self.train_cnn_after
            for param in self.model.cnn.parameters():
                param.requires_grad = cnn_grad
            if self.cnn_eval:
                self.model.cnn.eval()

        for idx , loader in enumerate(self.dataloaders):

            with tqdm(total=len(loader['loader'][self.phase]), desc="Batch processing", unit="Batch") as pbar:
                for ix_ , sample in enumerate(loader['loader'][self.phase]):

                    loss, loss_summands, batch_size = self.fit_sample(
                        sample,
                        grad_clip=self.grad_clip
                    )

                    running_losses[loader['name']] += loss * batch_size
                    running_loss_summands[loader['name']] = [
                        r + l * batch_size
                        for r, l in zip(running_loss_summands[loader['name']], loss_summands)
                    ]
                    n_samples[loader['name']] += batch_size
                    pbar.update(1)


        for idx , loader in enumerate(self.dataloaders):
            phase_loss = running_losses[loader['name']] / n_samples[loader['name']]
            phase_loss_summands = [
                loss_ / n_samples[loader['name']] for loss_ in running_loss_summands[loader['name']]
            ]

            print(
                f"{loader['name']:9s}:   Phase: {self.phase}, loss: {phase_loss:.4f}, "
                + ", ".join(
                    f"loss_{idx}: {loss_:.4f}"
                    for idx, loss_ in enumerate(phase_loss_summands)
                )
            )


            if (
                self.phase == "val"
                and self.epoch >= self.chkpnt_warmup
            ):
                val_score = -phase_loss
                if self.best_val_score is None:
                    self.best_val_score = val_score
                elif val_score > self.best_val_score:
                    print("UPADTE BEST WEIGHTS")
                    self.best_val_score = val_score
                    self.model.save_weights(self.train_dir, "best")
                    with open(self.train_dir / "best_epoch.dat", "w") as f:
                        f.write(str(self.epoch))
                    with open(self.train_dir / "best_val_loss.dat", "w") as f:
                        f.write(str(val_score))


    def fit_sample(self, sample, grad_clip=None):
        """
        Take a sample containing a batch, and fit/evaluate the model
        """

        with torch.set_grad_enabled(self.phase == "train"):
            x, sal, fix, _ = sample

            # Add temporal dimension to image data
            if x.dim() == 4:
                x = x.unsqueeze(1)
                sal = sal.unsqueeze(1)
                fix = fix.unsqueeze(1)
            x = x.float().to(self.device)
            sal = sal.float().to(self.device)
            fix = fix.to(self.device)

            if self.phase == "train":

                # Switch the RNN gradients off if this is a image batch
                rnn_grad = x.shape[1] != 1 or not self.model.bypass_rnn
                for param in chain(
                    self.model.rnn.parameters(), self.model.post_rnn.parameters()
                ):
                    param.requires_grad = rnn_grad

                # Switch the gradients of unused dataset-specific modules off
                for name, param in self.model.named_parameters():
                    # for source in self.all_data_sources:
                    #     if source.lower() in name.lower():
                    param.requires_grad = True

            # Run forward pass
            pred_seq = self.model(x)
            # pred_seq = self.model(x)

            # print("fix shape : " , fix.shape)
            # Compute the total loss
            loss_summands = self.loss_sequences(
                pred_seq, sal, fix, metrics=self.loss_metrics
            )
            # print("loss_summands " , loss_summands)
            loss_summands = [l.mean(1).mean(0) for l in loss_summands]
            loss = sum(
                weight * l for weight, l in zip(self.loss_weights, loss_summands)
            )

            # Run backward pass and optimization step
        if self.phase == "train":
            self.optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

        return loss.item(), [l.item() for l in loss_summands], x.shape[0]

    @staticmethod
    def loss_sequences(pred_seq, sal_seq, fix_seq, metrics):
        """
        Compute the training losses
        """

        losses = []
        for this_metric in metrics:
            if this_metric == "kld":
                losses.append(utils.kld_loss(pred_seq, sal_seq))
            if this_metric == "nss":
                losses.append(utils.nss(pred_seq.exp(), fix_seq))
            if this_metric == "cc":
                losses.append(utils.corr_coeff(pred_seq.exp(), sal_seq))
        return losses


    def get_model_parameter_groups(self):
        """
        Get parameter groups.
        Output CNN parameters separately with reduced LR and weight decay.
        """

        def parameters_except_cnn():
            parameters = []
            adaptation = []
            for name, module in self.model.named_children():
                if name == "cnn":
                    continue
                elif "adaptation" in name:
                    adaptation += list(module.parameters())
                else:
                    parameters += list(module.parameters())
            return parameters, adaptation

        parameters, adaptation = parameters_except_cnn()

        for name, this_parameter in self.model.named_parameters():
            if "gaussian" in name:
                parameters.append(this_parameter)

        return [
            {"params": parameters + adaptation},
            {
                "params": self.model.cnn.parameters(),
                "lr": self.lr * self.cnn_lr_factor,
                "weight_decay": self.cnn_weight_decay,
            },
        ]
    

    def save_chkpnt(self):
        """Save model and trainer checkpoint"""
        print(f"Saving checkpoint at epoch {self.epoch}")
        chkpnt = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        torch.save(chkpnt, self.train_dir / 'checkpoints' / f"chkpnt_epoch{self.epoch:04d}.pth")


    @property
    def optimizer(self):
        """Return the optimizer"""
        if self._optimizer is None:
            if self.optim_algo == "SGD":
                self._optimizer = torch.optim.SGD(
                    self.get_model_parameter_groups(),
                    lr=self.lr,
                    momentum=self.momentum,
                    weight_decay=self.weight_decay,
                )
            elif self.optim_algo == "Adam":
                self._optimizer = torch.optim.Adam(
                    self.get_model_parameter_groups(),
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                )
            elif self.optim_algo == "RMSprop":
                self._optimizer = torch.optim.RMSprop(
                    self.get_model_parameter_groups(),
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                    momentum=self.momentum,
                )

        return self._optimizer

    @property
    def scheduler(self):
        """Return the learning rate scheduler"""
        if self._scheduler is None:
            if self.lr_scheduler == "ExponentialLR":
                self._scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer, gamma=self.lr_gamma, last_epoch=self.epoch - 1
                )
            else:
                raise ValueError(f"Unknown scheduler {self.lr_scheduler}")
        return self._scheduler



import argparse

if __name__ == "__main__":

    # Créez un parser d'arguments
    parser = argparse.ArgumentParser(description='Trainer for the model.')

    # Ajoutez les arguments pour le Trainer
    parser.add_argument('--num_epochs', type=int, default=100, help='Nombre d\'époques pour l\'entraînement.')
    parser.add_argument('--optim_algo', type=str, default="SGD", help='Algorithme d\'optimisation.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum pour l\'optimiseur.')
    parser.add_argument('--lr', type=float, default=0.04, help='Taux d\'apprentissage.')
    parser.add_argument('--lr_scheduler', type=str, default="ExponentialLR", help='Type de planificateur de taux d\'apprentissage.')
    parser.add_argument('--lr_gamma', type=float, default=0.99, help='Facteur gamma pour le planificateur de taux d\'apprentissage.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Poids de décroissance pour l\'optimiseur.')
    parser.add_argument('--cnn_weight_decay', type=float, default=1e-5, help='Poids de décroissance pour le CNN.')
    parser.add_argument('--grad_clip', type=float, default=2.0, help='Valeur de coupure de gradient.')
    parser.add_argument('--cnn_lr_factor', type=float, default=0.1, help='Facteur de taux d\'apprentissage pour le CNN.')
    parser.add_argument('--loss_metrics', type=str, nargs='+', default=["kld", "nss", "cc"], help='Métriques de perte à utiliser.')
    parser.add_argument('--loss_weights', type=float, nargs='+', default=[1, -0.1, -0.1], help='Poids des métriques de perte.')
    parser.add_argument('--chkpnt_warmup', type=int, default=2, help='Époques de montée en température pour le point de contrôle.')
    parser.add_argument('--chkpnt_epochs', type=int, default=2, help='Nombre d\'époques pour sauvegarder le point de contrôle.')
    parser.add_argument('--path_save', type=str, default="./weights/video_test/" , help='path save output')
    parser.add_argument('--path_dataset', type=str, default="C:/Users/Shadow/Documents/Dataset/Packaging_delta_1_sigma_20/" , help='path dataset')

    # Analysez les arguments
    args = parser.parse_args()

    # create model Unisal
    unisal_ = model.UNISAL(bypass_rnn=False)

    # load model from github and res
    directory_ = "./weights/weights_best.pth"
    unisal_.load_weights(directory_ )
    # move model to device
    print(f"Move model to torch device set to: {DEFAULT_DEVICE}")
    unisal_.to(DEFAULT_DEVICE)

    packaging_train = dataloaders.PACKAGINGDataset(path="/Users/coconut/Documents/Dataset/GenSaliency/VisualSaliency/Packaging_delta_3_sigma_20/" + "/train/")
    packaging_val = dataloaders.PACKAGINGDataset(path="/Users/coconut/Documents/Dataset/GenSaliency/VisualSaliency/Packaging_delta_3_sigma_20/" + "/val/")

    video_train = dataloaders.VideoDataset(path= "/Users/coconut/Documents/Dataset/GenSaliency/VisualSaliency/ittention_videos/" + "/train/", N = 12)
    video_val = dataloaders.VideoDataset(path="/Users/coconut/Documents/Dataset/GenSaliency/VisualSaliency/ittention_videos/" + "/val/", N=12)

    # print("Len Dataset : {}".format(len(packaging_)))

    dataloaders_ = [
    {
        'name' : 'Packaging',
        'loader' : {
            'train' : DataLoader(packaging_train, batch_size=10, shuffle=True),
            'val' : DataLoader(packaging_val, batch_size=10, shuffle=True)
        }
    },
    {
        'name' : 'Video',
        'loader' : {
            'train' : DataLoader(video_train, batch_size=4, shuffle=True),
            'val' : DataLoader(video_val, batch_size=4, shuffle=True)
        }

    }
    ]


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
    )

    trainer.fit()


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
import utils.utils as utils
import torch
import json
from itertools import zip_longest
from random import shuffle

if torch.cuda.is_available():
    DEFAULT_DEVICE = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    DEFAULT_DEVICE = torch.device("mps")
else:
    DEFAULT_DEVICE = torch.device("cpu")


import sys
import time

# Fonction pour afficher la barre de progression avec timer
def loading_bar(total, current, start_time,text = "Batches", bar_length=30):
    percent = current / total
    arrow = '█' * int(percent * bar_length)
    spaces = ' ' * (bar_length - len(arrow))

    # Calcule le temps écoulé
    elapsed_time = time.time() - start_time
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    # Affiche la barre de progression avec le timer
    sys.stdout.write(f'\r{text}: |{arrow}{spaces}| {int(percent * 100)}% - timer: {elapsed_str}')
    sys.stdout.flush()

class Trainer():
    def __init__(
        self,
        dataloaders,
        device,
        model ,
        path ,

        num_epochs=50,
        optim_algo="SGD",
        momentum=0.9,
        lr=0.04,
        lr_scheduler="ExponentialLR",
        lr_gamma=0.99,
        weight_decay=1e-4,
        cnn_weight_decay=1e-5,
        grad_clip=1.0,
        cnn_lr_factor=0.1,
        loss_metrics=("kld", "nss", "cc"),
        loss_weights=(1, -0.1, -0.1),
        chkpnt_warmup=2,
        chkpnt_epochs=2,
        train_cnn_after= 10,

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
        self.train_cnn_after = train_cnn_after
        self.cnn_eval = True
        self.loss_weights = (1,-0.1 , -0.1)

        self.train_dir = Path(path)
        if os.path.exists(path) == False:
            os.mkdir(self.train_dir)

        # save sources informations
        model_sources_ = [src for src in model.sources]
        json_object = json.dumps(model_sources_, indent=4)
        with open(path + "sources.json", "w") as outfile:
            outfile.write(json_object)

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

        score_ = 1e8
        while self.epoch < self.num_epochs:
            self.scheduler.step(epoch=self.epoch)
            lr = self.optimizer.param_groups[0]["lr"]
            print(f"\nEpoch {self.epoch:3d}, lr {lr:.5f}")

            losses_ = {
                'epoch' : self.epoch
            }

            for self.phase in self.phases:
                losses_[self.phase] = self.fit_phase()

            # save best weights
            if score_ > losses_['val']['global']:
                score_ = losses_['val']['global']
                self.save_best(losses_)

            # Save a checkpoint if applicable
            if (
                self.epoch >= self.chkpnt_warmup
                and (self.epoch + 1) % self.chkpnt_epochs == 0
            ) or self.epoch == self.num_epochs - 1:
                self.save_chkpnt(losses_)

            self.epoch += 1

    def fit_phase(self):
        """
        Run the current phase (training or validation)
        """

        # Prepare book keeping
        running_losses = {key: 0.0 for key , _ in self.dataloaders.items()}
        running_eval = {key: 0.0 for key, _ in self.dataloaders.items()}
        running_loss_summands = {
            key: [0.0 for _ in self.loss_weights] for key, _ in self.dataloaders.items()
        }
        n_samples = {key: 0 for key, _ in self.dataloaders.items()}

        all_batches = [
            src
            for src in chain.from_iterable(
                zip_longest(
                    *[[key for _ in range(len(v[self.phase]))] for key, v in self.dataloaders.items()]
                )
            )
            if src is not None
        ]

        shuffle(all_batches)

        if self.epoch == 0:
            print(f"Number of batches: {len(all_batches)}")
            print(", ".join(f"{key}: {len(v[self.phase])}" for key, v in self.dataloaders.items()))


        # Set model train/eval mode
        self.model.train(self.phase == "train")

        # Switch CNN gradients on/off and set CNN eval mode (for BN modules)
        if self.phase == "train":
            cnn_grad = self.epoch >= self.train_cnn_after
            for param in self.model.cnn.parameters():
                param.requires_grad = cnn_grad
            if self.cnn_eval:
                self.model.cnn.eval()


        start_time = time.time()
        data_iters = {key: iter(v[self.phase]) for key, v in self.dataloaders.items()}
        for sample_idx, src in enumerate(all_batches):

            loading_bar(text = f"{self.phase} Batches ",current=sample_idx ,total=len(all_batches),bar_length=40, start_time=start_time)
            # Get the next batch
            sample = next(data_iters[src])

            loss, loss_summands, batch_size = self.fit_sample(
                src,
                sample,
                grad_clip=self.grad_clip
            )

            running_losses[src] += loss * batch_size
            running_loss_summands[src] = [
                r + l * batch_size
                for r, l in zip(running_loss_summands[src], loss_summands)
            ]
            n_samples[src] += batch_size

        print()


        loss_object_ = {
            'global' : 0.
        }
        for key , v in self.dataloaders.items():
            phase_loss = running_losses[key] / n_samples[key]
            phase_loss_summands = [
                loss_ / n_samples[key] for loss_ in running_loss_summands[key]
            ]

            loss_object_[key] = {
                'loss' : phase_loss,
                'losses' : {}
            }

            print(
                f"  - {key:9s}:   Phase: {self.phase}, loss: {phase_loss:.4f}, "
                + ", ".join(
                    f"loss {idx}: {loss_:.4f}"
                    for idx, loss_ in zip(self.loss_metrics, phase_loss_summands)
                )
            )

            for idx, loss_ in zip(self.loss_metrics, phase_loss_summands):
                loss_object_[key]['losses'][idx] =loss_

            loss_object_['global'] += phase_loss
        loss_object_['global'] /= len(self.dataloaders)

        return loss_object_

    def fit_sample(self,loader_name, sample, grad_clip=None):
        """
        Take a sample containing a batch, and fit/evaluate the model
        """

        with torch.set_grad_enabled(self.phase == "train"):
            x, sal, fix, target_size = sample

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
                    for source in self.model.sources:
                        if source.lower() in name.lower():
                            param.requires_grad = source == loader_name

            # Run forward pass
            pred_seq = self.model(
                x = x,
                source = loader_name,
                target_size = target_size
                )

            
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

    def save_best(self,losses):
        """Save best model and losses"""
        print("      - UPADTE BEST WEIGHTS")
        self.model.save_weights(self.train_dir, "best")
        with open(self.train_dir / "best_epoch.dat", "w") as f:
            f.write(str(self.epoch))

        json_object = json.dumps(losses, indent=4)
        with open(self.train_dir / "best_epoch_loss.json" , "w") as f:
            f.write(json_object)
    

    def save_chkpnt(self, losses, save_weights = False):
        """Save model and trainer checkpoint"""
        print(f"Saving checkpoint at epoch {self.epoch}")
        chkpnt = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        if save_weights:
            torch.save(chkpnt, self.train_dir / 'checkpoints' / f"chkpnt_epoch{self.epoch:04d}.pth")

        json_object = json.dumps(losses, indent=4)
        with open( self.train_dir / 'checkpoints' / f"chkpnt_epoch{self.epoch:04d}.json" , "w") as f:
            f.write(json_object)


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


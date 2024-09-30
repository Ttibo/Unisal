# Trainer for UNISAL Model

Implementation simplifié du modèle UNISAL pour la saillance, en utilisant PyTorch. Ce code inclut les fonctionnalités de formation et de validation, ainsi que la gestion des checkpoints.

## Table des matières

- [Fonctionnalités](#fonctionnalités)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Arguments](#arguments)
- [Bases de données](#Bases-de-données)
- [Exemple d'utilisation](#exemple-dutilisation)
- [Contributions](#contributions)

## Fonctionnalités

- Chargement et entraînement d'un modèle UNISAL.
- Prise en charge de l'entraînement et de la validation.
- Sauvegarde des checkpoints à intervalles réguliers.
- Personnalisation des hyperparamètres via les arguments de ligne de commande.
- Export Onnx model

## Installation

Assurez-vous d'avoir Python et PyTorch installés sur votre système. Vous pouvez installer les dépendances nécessaires avec pip :

```bash
pip install -r requirements.txt
```

### Dépendances
- Python 3.11.9
- PyTorch
- OpenCV
- Matplotlib
- NumPy

## Utilisation

Pour exécuter le script d'entraînement, utilisez la commande suivante dans le terminal :

```bash
python trainer.py --num_epochs 100 --path_save ./weights/
```

Vous pouvez ajuster les paramètres en fonction de vos besoins.

## Arguments

Voici les arguments que vous pouvez utiliser pour personnaliser l'entraînement :

| Argument               | Type     | Description                                          | Valeur par défaut           |
|-----------------------|----------|----------------------------------------------------|-----------------------------|
| `--num_epochs`        | `int`    | Nombre d'époques pour l'entraînement.              | `100`                       |
| `--optim_algo`        | `str`    | Algorithme d'optimisation.                         | `"SGD"`                     |
| `--momentum`          | `float`  | Momentum pour l'optimiseur.                        | `0.9`                       |
| `--lr`                | `float`  | Taux d'apprentissage.                              | `0.04`                      |
| `--lr_scheduler`      | `str`    | Type de planificateur de taux d'apprentissage.     | `"ExponentialLR"`           |
| `--lr_gamma`          | `float`  | Facteur gamma pour le planificateur de taux.      | `0.99`                      |
| `--weight_decay`      | `float`  | Poids de décroissance pour l'optimiseur.          | `1e-4`                      |
| `--cnn_weight_decay`  | `float`  | Poids de décroissance pour le CNN.                 | `1e-5`                      |
| `--grad_clip`         | `float`  | Valeur de coupure de gradient.                     | `2.0`                       |
| `--cnn_lr_factor`     | `float`  | Facteur de taux d'apprentissage pour le CNN.      | `0.1`                       |
| `--loss_metrics`      | `str[]`  | Métriques de perte à utiliser.                     | `["kld", "nss", "cc"]`     |
| `--loss_weights`      | `float[]`| Poids des métriques de perte.                      | `[1, -0.1, -0.1]`          |
| `--chkpnt_warmup`     | `int`    | Époques de montée en température pour le point de contrôle. | `2`                  |
| `--chkpnt_epochs`     | `int`    | Nombre d'époques pour sauvegarder le point de contrôle. | `2`                  |
| `--path_save`         | `str`    | Chemin pour sauvegarder les poids du modèle.      | `./weights/packging_1s/`   |


## Bases de données
Structure donnes pour entrainement

- x : Un tenseur contenant les images d'entrée (format : [batch_size, channels, height, width])
- sal : Un tenseur contenant les cartes de salience correspondantes (format : [batch_size, channels, height, width])
- fix : Un tenseur contenant les points de fixation (format : [batch_size, channels, height, width])
- target_size : La taille cible des images, utilisée pour le redimensionnement

![Exemple 1](ressources/exemple_1.png)
![Exemple 2](ressources/exemple_2.png)



## Exemple d'utilisation

Voici un exemple d'utilisation :

```bash
python trainer.py --num_epochs 50 --optim_algo Adam --lr 0.001 --path_save ./output/
```

## Contributions

Les contributions sont les bienvenues ! Veuillez soumettre une demande de tirage (pull request) ou ouvrir une issue pour discuter des améliorations potentielles.


# TODO 
- [ ] Gestion dataloaser train/val 
- [ ] Clean class
- [ ] Modification modèle encoder
- [ ] Entrainement et dataloaders video

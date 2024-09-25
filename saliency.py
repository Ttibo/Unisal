import os
import math
import cv2
import matplotlib.pyplot as plt
from .model import UNISAL
from .dataloaders import *
from .utils import *
import yaml

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union


import torch
if torch.cuda.is_available():
    DEFAULT_DEVICE = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    DEFAULT_DEVICE = torch.device("mps")
else:
    DEFAULT_DEVICE = torch.device("cpu")

class Saliency:
    def __init__(self , modelName : str = "unisal" , pathModel : str = "/weights/weights_best.pth"):
        assert(modelName in ['unisal']) , 'Error model choices [unisal]'
        # assert(os.path.isfile(pathModel) == True) , 'Error path model, file doesn\'t exist'

        self.path_ = os.path.dirname(os.path.abspath(__file__))
        with open(self.path_ + '/helper/colors.yaml', "r") as stream:
            self.colors = yaml.safe_load(stream)
        
        if modelName == 'unisal':
            print("load model")
            self.model = UNISAL(bypass_rnn=True)

        print("PathModel " , pathModel)
        self.model.load_weights(self.path_ + pathModel)
        self.model.to(DEFAULT_DEVICE)

    def normalize(self , image : np.ndarray ) -> np.ndarray: 
        # normalize image btw 0 and 1 : -> float32
        print("normalize : " ,image.shape)

        return cv2.normalize(image, image, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    
    def show_maps(self , img , colormap ,map):
        # Créer une figure avec 4 sous-graphiques
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Afficher la première image
        axs[0].imshow(img.numpy().transpose(1,2,0))
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

        # Afficher le tout
        plt.show()

    def run(self , pathImage : str) -> np.ndarray:

        img = cv2.imread(str(pathImage))
        img_ , size_ = open_image(pathImage)
        map_ = self.image_inference(img_ )
        
        smap = map_.exp()
        smap = torch.squeeze(smap)
        smap = smap.detach().cpu().numpy()
        map_ = (smap / np.amax(smap) * 255).astype(np.uint8)
        
        colormap_ = self.get_color(img_ , map_)

        predicted_colored = colormap_.astype(np.uint8)
        predicted_colored = cv2.resize(predicted_colored , (img.shape[1] , img.shape[0]))
        resultat = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0.3, predicted_colored, 0.7, 0.0)

        return map_ , colormap_ , resultat

    def image_inference(self , img : torch.Tensor ) -> np.ndarray :
        img_ = img.to(DEFAULT_DEVICE).unsqueeze(0).unsqueeze(0)
        map_ = self.model(img_)
        return map_.squeeze(0).squeeze(0).squeeze(0)
    
    def get_color(self,image_ : np.ndarray, map_ : np.ndarray , exposant: int = 2) -> np.ndarray:
        """Return the saliencymap as a 4 color heatmap
        Args:
            exposant (int): power value applied on the saliency map (default: {2})
        Returns:
            np.ndarray: 4 colors heatmap
            None: if no saliency map has been previously computed
        """

        # Different heat maps with green colors and different exposants
        # Heat map with green colors
        non_linear_heat_green_cmap = np.array(
            [
                self.colors["ITT_HEAT_GREEN"]["B"],
                self.colors["ITT_HEAT_GREEN"]["G"],
                self.colors["ITT_HEAT_GREEN"]["R"],
            ]
        ).swapaxes(1, 0)
        alpha = 0.7
        height, width = map_.shape[:2]

        # val_map_heat_green = np.power(map_, exposant)
        # val_map_heat_green = np.clip(255 * val_map_heat_green, 0, 255).astype(int)

        # 2D to 1D array(easier for next line)
        saliency_index_green = map_.reshape(-1, 1)
        heat_map_green = non_linear_heat_green_cmap[saliency_index_green]
        heat_map_green = heat_map_green.reshape((height, width, 3))
        heat_map_green = alpha * heat_map_green 

        return heat_map_green
    




        
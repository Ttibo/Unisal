import sys
sys.path.append("./")
sys.path.append("../")

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

import model
import dataloaders
import json

if torch.cuda.is_available():
    DEFAULT_DEVICE = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    DEFAULT_DEVICE = torch.device("mps")
else:
    DEFAULT_DEVICE = torch.device("cpu")


class Saliency:
    def __init__(self  , pathModel : str ):
        # assert(os.path.isfile(pathModel) == True) , 'Error path model, file doesn\'t exist'

        assert( os.path.exists(pathModel)) , " Error folder model weights"

        if os.path.isfile(pathModel + "sources.json"):
            with open(pathModel + "sources.json", 'r') as file:
                sources = json.load(file) 
            self.model = model.UNISAL(bypass_rnn=False, sources=sources)
        else:
            self.model = model.UNISAL(bypass_rnn=False)

        self.model.load_weights(pathModel + "weights_best.pth")
        self.model.to(DEFAULT_DEVICE)

        # Charger le modèle ONNX avec ONNX Runtime
        # self.session = ort.InferenceSession(self.path_ + pathModelOnnx, providers=[DEFAULT_DEVICE_ONNX])


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


    def run(self , pathImage : str ) -> np.ndarray:

        img = cv2.imread(str(pathImage))
        img_ , size_ = dataloaders.open_image(pathImage)
        map_ = self.image_inference(img_ )

        smap = np.exp(map_)
        smap = np.squeeze(smap)
        smap = smap
        map_ = (smap / np.amax(smap) * 255).astype(np.uint8)
        map_ = cv2.resize(map_ , (img.shape[1] , img.shape[0]))
        
        predicted_colored = cv2.applyColorMap(map_.astype(np.uint8) , cv2.COLORMAP_JET)
        res_ = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0.3, cv2.cvtColor(predicted_colored, cv2.COLOR_BGR2RGB), 0.7, 0.0)

        self.show_maps(img , map_ , res_)

    def image_inference(self , img : torch.Tensor ) -> np.ndarray :
        img_ = img.to(DEFAULT_DEVICE).unsqueeze(0).unsqueeze(0)
        print(img.dtype)

        map_ = self.model(img_, source="SALICON")
        return map_.squeeze(0).squeeze(0).squeeze(0).detach().cpu().numpy()
    


import argparse
if __name__ == "__main__":

    # Créez un parser d'arguments
    parser = argparse.ArgumentParser(description='Trainer for the model.')

    # Ajoutez les arguments pour le Trainer
    parser.add_argument('--image', type=str,default= "./inputs/test_1.jpg",  help='path image')
    parser.add_argument('--model', type=str,default= "../model/weights/",  help='path model')

    args = parser.parse_args()

    saliency_ = Saliency(pathModel = args.model )
    saliency_.run(args.image)

    plt.show()

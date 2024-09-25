import os
import cv2
import matplotlib.pyplot as plt
import model
import numpy as np
import torch
import dataloaders

if torch.cuda.is_available():
    DEFAULT_DEVICE = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    DEFAULT_DEVICE = torch.device("mps")
else:
    DEFAULT_DEVICE = torch.device("cpu")

class Saliency:
    def __init__(self , modelName : str = "unisal" , pathModel : str = "/weights/packging_3s/weights_best.pth"):
        assert(modelName in ['unisal']) , 'Error model choices [unisal]'
        # assert(os.path.isfile(pathModel) == True) , 'Error path model, file doesn\'t exist'

        self.path_ = os.path.dirname(os.path.abspath(__file__))
        if modelName == 'unisal':
            print("load model")
            self.model = model.UNISAL(bypass_rnn=True)

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

        # Afficher le tout
        plt.show()

    def run(self , pathImage : str) -> np.ndarray:

        img = cv2.imread(str(pathImage))
        img_ , size_ = dataloaders.open_image(pathImage)
        map_ = self.image_inference(img_ )
        
        smap = map_.exp()
        smap = torch.squeeze(smap)
        smap = smap.detach().cpu().numpy()
        map_ = (smap / np.amax(smap) * 255).astype(np.uint8)
        map_ = cv2.resize(map_ , (img.shape[1] , img.shape[0]))

        
        predicted_colored = cv2.cvtColor(map_.astype(np.uint8) , cv2.COLORMAP_JET)
        res_ = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0.3, cv2.cvtColor(predicted_colored, cv2.COLOR_BGR2RGB), 0.7, 0.0)

        self.show_maps(img , map_ , res_)

    def image_inference(self , img : torch.Tensor ) -> np.ndarray :
        img_ = img.to(DEFAULT_DEVICE).unsqueeze(0).unsqueeze(0)
        map_ = self.model(img_)
        return map_.squeeze(0).squeeze(0).squeeze(0)
    

if __name__ == "__main__":

    saliency_ = Saliency(pathModel = "/weights/packging_3s/weights_best.pth")
    saliency_.run("/Users/thibaultlelong/Documents/Dataset/GenSaliency/test/image_1.jpg")

        
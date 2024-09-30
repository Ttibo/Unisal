

import os
import PIL
import cv2
import torch.onnx
import numpy as np
from model import UNISAL
import onnxruntime as ort
from torchvision import transforms
import matplotlib.pyplot as plt


# Dispositif par défaut - CPU ou GPU pour ONNX
if ort.get_device() == 'GPU':
    DEFAULT_DEVICE = 'gpu'
else:
    DEFAULT_DEVICE = 'cpu'


class SaliencyONNX:
    _preproc_cfg = {
        'rgb_mean': (0.485, 0.456, 0.406),
        'rgb_std': (0.229, 0.224, 0.225),
    }

    def __init__(self) -> None:
        
        path_ = os.path.dirname(os.path.abspath(__file__))

        # Charger le modèle ONNX en fonction du modèle sélectionné
        self.model_path = os.path.join(path_, "weights/unisal_model.onnx")

        # Configurer le fournisseur ONNX pour GPU ou CPU
        if DEFAULT_DEVICE == 'gpu':
            self.session = ort.InferenceSession(self.model_path, providers=['CUDAExecutionProvider'])
        else:
            self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])

    def get_optimal_out_size(self, img_size: tuple) -> tuple:
        """Calculate the optimal output size for the given image size."""
        aspect_ratio = img_size[0] / img_size[1]
        min_product = 100
        max_product = 120
        aspect_ratios = []
        size_array = []

        for n1 in range(7, 14):
            for n2 in range(7, 14):
                if min_product <= n1 * n2 <= max_product:
                    current_aspect_ratio = n1 / n2
                    aspect_ratio_diff = min((aspect_ratio, current_aspect_ratio)) / max((aspect_ratio, current_aspect_ratio))
                    aspect_ratios.append(aspect_ratio_diff)
                    size_array.append((n1, n2))

        max_aspect_ratio_idx = np.argmax(np.array(aspect_ratios)).item()
        optimal_size = tuple(dim * 32 for dim in size_array[max_aspect_ratio_idx])
        return optimal_size

    def preprocess(self, img: np.ndarray, out_size: tuple) -> np.ndarray:
        """Preprocess the image for the ONNX model."""
        transformations = [
            transforms.ToPILImage(),
            transforms.Resize(out_size, interpolation=PIL.Image.LANCZOS),
            transforms.ToTensor(),
        ]

        if 'rgb_mean' in self._preproc_cfg:
            transformations.append(
                transforms.Normalize(self._preproc_cfg['rgb_mean'], self._preproc_cfg['rgb_std'])
            )

        processing = transforms.Compose(transformations)
        tensor = processing(img)
        return tensor.numpy()  # Convertir le tensor en numpy array pour ONNX

    def open_image(self, file: str) -> tuple:
        """Open an image file and preprocess it."""
        img = cv2.imread(str(file))
        assert img is not None, "Error: Unable to open image. Verify your file."

        img = np.ascontiguousarray(img[:, :, ::-1])  # Convert BGR to RGB
        out_size = self.get_optimal_out_size(img.shape[:2])
        out_img = self.preprocess(img, out_size)
        return out_img, tuple(img.shape[:2])

    def run(self, path_image: str) -> np.ndarray:
        """Run inference on the provided image path using ONNX."""

        img = cv2.imread(path_image)
        img_tensor, original_size = self.open_image(path_image)

        print(img_tensor.shape)

        map_ = self.image_inference(img_tensor)

        # Post-process saliency map
        saliency_map = np.exp(map_)
        saliency_map = saliency_map.squeeze()
        saliency_map /= np.amax(saliency_map)  # Normalize
        saliency_map = cv2.resize(saliency_map, (img.shape[1], img.shape[0]))

        predicted_colored = cv2.cvtColor(saliency_map.astype(np.uint8) , cv2.COLORMAP_JET)
        res_ = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0.3, cv2.cvtColor(predicted_colored, cv2.COLOR_BGR2RGB), 0.7, 0.0)

        self.show_maps(img , saliency_map , res_)
        return saliency_map

    def image_inference(self, img: np.ndarray) -> np.ndarray:
        """Perform inference using the ONNX model."""
        img_ = img[np.newaxis, np.newaxis, ...]  # Add batch and channel dimensions
        print(img_.shape)
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        ort_inputs = {input_name: img_}
        ort_outs = self.session.run([output_name], ort_inputs)
        return ort_outs[0].squeeze()  # Remove batch and channel dimensions
    
        
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


if __name__ == "__main__":

    # Supposons que votre modèle soit déjà instancié
    model = UNISAL()
    model.load_weights("./weights/packging_3s/weights_best.pth")

    # Assurez-vous que le modèle est en mode évaluation (très important pour ONNX)
    model.eval()

    # Exemple d'entrée de taille [batch_size, time_steps, channels, height, width]
    # Vous devez adapter la taille en fonction de vos besoins
    example_input = torch.randn(1 , 1, 3, 256, 416)  # Batch size 1, 3 time steps, 3 channels, 224x224 image

    # Chemin du fichier de sortie
    onnx_file_path = "./weights/packging_3s/unisal_model.onnx"

    # Exportation en ONNX avec des dimensions dynamiques
    torch.onnx.export(
        model,                        # Modèle
        example_input,                # Entrée factice
        onnx_file_path,               # Nom du fichier de sortie
        export_params=True,           # Exporter aussi les paramètres du modèle
        opset_version=12,             # Version de l'IR (ONNX 11 est compatible avec la plupart des frameworks)
        do_constant_folding=True,     # Activer ou désactiver le repliement constant
        input_names=['input'],        # Nom des entrées
        output_names=['output'],      # Nom des sorties
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'time_steps', 3: 'height', 4: 'width'},  # Dimensions dynamiques pour l'entrée
            'output': {0: 'batch_size', 1: 'time_steps', 3: 'height', 4: 'width'}  # Dimensions dynamiques pour la sortie
        }
    )

    print(f"Modèle exporté en ONNX vers {onnx_file_path}")
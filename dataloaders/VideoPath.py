import os
from torchvision import transforms
import numpy as np
import cv2
import PIL
import torch
from tqdm import tqdm

class VideoPath:

    def __init__(self):
        self.preproc_cfg = {
            'rgb_mean': (0.485, 0.456, 0.406),
            'rgb_std': (0.229, 0.224, 0.225),
        }

    def get_optimal_out_size(self, img_size):
        ar = img_size[0] / img_size[1]
        min_prod = 100
        max_prod = 120  
        ar_array = []
        size_array = []
        for n1 in range(7, 14):
            for n2 in range(7, 14):
                if min_prod <= n1 * n2 <= max_prod:
                    this_ar = n1 / n2
                    this_ar_ratio = min((ar, this_ar)) / max((ar, this_ar))
                    ar_array.append(this_ar_ratio)
                    size_array.append((n1, n2))

        max_ar_ratio_idx = np.argmax(np.array(ar_array)).item()
        bn_size = size_array[max_ar_ratio_idx]
        out_size = tuple(r * 32 for r in bn_size)
        return out_size

    def preprocess(self, img, out_size):
        transformations = [
            transforms.ToPILImage(),
            transforms.Resize(
                out_size, interpolation=PIL.Image.LANCZOS),
            transforms.ToTensor(),
        ]
        if 'rgb_mean' in self.preproc_cfg:
            transformations.append(
                transforms.Normalize(
                    self.preproc_cfg['rgb_mean'], self.preproc_cfg['rgb_std']))
        processing = transforms.Compose(transformations)
        tensor = processing(img)
        return tensor
    
    def open_video(self, file: str, fps: int):
        assert(os.path.isfile(file)) , f"Error : file {file} not found... "
        cap = cv2.VideoCapture(file)

        # Vérifier si la vidéo s'est ouverte correctement
        assert(cap.isOpened) ,"Erreur : Impossible d'ouvrir la vidéo"
        self.frames = []
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"FrameRate {original_fps}")

        # Calculer le ratio entre le framerate d'origine et le nouveau framerate
        if fps < original_fps:
            fps_ratio = original_fps / fps
        else:
            fps_ratio = 1

        # Lire la vidéo frame par frame
        frame_index = 0
        while cap.isOpened():
            # Calculer l'index de la prochaine frame à lire
            frame_index += fps_ratio

            # Aller à la frame correspondante
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))

            ret, frame = cap.read()  # Lire la frame sélectionnée
            if not ret:
                print("Fin de la vidéo ou erreur de lecture")
                break
            self.frames.append(frame)

        print(f"Number Frames -> {len(self.frames)}")
        self.frame_index = 0

    def get_frame(self):
        if len(self.frames) == self.frame_index:
            print("Video Done ....")
            return None
        img = self.frames[self.frame_index]        
        img = np.ascontiguousarray(img[:, :, ::-1])
        out_size = self.get_optimal_out_size(tuple(img.shape[:2]))
        out_img = self.preprocess(img, out_size)
        self.frame_index += 1

        return (out_img, tuple(img.shape[:2]))
    

    def get_packages_of_frames(self, size: int = 4):
        packages_ = None
        for i in range(self.frame_index , min(len(self.frames) , self.frame_index + size )):
            img = self.frames[i]        
            img = np.ascontiguousarray(img[:, :, ::-1])
            out_size = self.get_optimal_out_size(tuple(img.shape[:2]))
            out_img = self.preprocess(img, out_size).unsqueeze(0)

            if packages_ is None:
                packages_ = out_img
            else:
                packages_ = torch.cat([packages_, out_img], dim=0)

        self.frame_index += size
        return packages_




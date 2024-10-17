import sys
sys.path.append("./")
sys.path.append("../")

import os
import cv2
import matplotlib.pyplot as plt
import model
import numpy as np
import torch
from tqdm import tqdm
from dataloaders import VideoPath
import onnxruntime as ort  # Utilisation d'ONNX Runtime

if torch.cuda.is_available():
    DEFAULT_DEVICE = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    DEFAULT_DEVICE = torch.device("mps:0")
else:
    DEFAULT_DEVICE = torch.device("cpu")

if __name__ == "__main__":
    # open model
    model_path = "/weights/packging_3s/weights_best.pth"
    model_path = "/weights/video_test/weights_best.pth"
    model_path = "../weights/weights_best.pth"
    # model_path = "../weights/Train_packging_advertising/weights_best.pth"

    path_ = os.path.dirname(os.path.abspath(__file__))
    model = model.UNISAL(bypass_rnn=False)
    model.load_weights( model_path)
    model.to(DEFAULT_DEVICE)
    print(f"Device {DEFAULT_DEVICE}")
    frames_predic = []

    file_ = "/Users/coconut/Documents/Dataset/GenSaliency/test/video_0/ff.mp4"
    # file_ = "/Users/coconut/Documents/Video/Building makemore Part 2 MLP.mp4"
    videopath = VideoPath()
    videopath.open_video(
        file = file_ , 
        fps = 15
    )


    # hidden state for RNN video
    size_package = 4
    h0 = [None]
    total_frames = len(videopath.frames) 
    with tqdm(total=total_frames, desc="Traitement des frames", unit="frame") as pbar:
        while True:
            this_frame_seq =  videopath.get_packages_of_frames(size=size_package)

            if this_frame_seq is None:
                break

            # print(f"Package {this_frame_seq.shape}")
            this_frame_seq = this_frame_seq.to(DEFAULT_DEVICE).unsqueeze(0)

            this_pred_seq, h0 = model(
                this_frame_seq, return_hidden=True
            )

            pbar.update(this_pred_seq.shape[1])

            this_pred_seq = this_pred_seq.squeeze(0).detach().cpu().numpy()
            for i in range(0 , this_pred_seq.shape[0]):
                frames_predic.append(this_pred_seq[i])

            pbar.update(1)
            
    for i in range(0 , len(frames_predic)):

        map_ = frames_predic[i]
        img = videopath.frames[i]

        smap = np.exp(map_)
        smap = np.squeeze(smap)
        smap = smap
        map_ = (smap / np.amax(smap) * 255).astype(np.uint8)
        map_ = cv2.resize(map_ , (img.shape[1] , img.shape[0]))
        
        predicted_colored = cv2.applyColorMap(map_.astype(np.uint8) , cv2.COLORMAP_JET)
        res_ = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0.4, cv2.cvtColor(predicted_colored, cv2.COLOR_BGR2RGB), 0.7, 0.0)

        cv2.imshow("res" , res_)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

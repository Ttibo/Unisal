
from torchvision import transforms
import numpy as np
import cv2
import PIL

preproc_cfg = {
    'rgb_mean': (0.485, 0.456, 0.406),
    'rgb_std': (0.229, 0.224, 0.225),
}

def get_optimal_out_size(img_size):
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

def preprocess(img, out_size):
    transformations = [
        transforms.ToPILImage(),
        transforms.Resize(
            out_size, interpolation=PIL.Image.LANCZOS),
        transforms.ToTensor(),
    ]
    if 'rgb_mean' in preproc_cfg:
        transformations.append(
            transforms.Normalize(
                preproc_cfg['rgb_mean'], preproc_cfg['rgb_std']))
    processing = transforms.Compose(transformations)
    tensor = processing(img)
    return tensor

def open_image(file):
    img = cv2.imread(str(file))
    assert (img is not None) , "Error open Image. Verify your file"
    img = np.ascontiguousarray(img[:, :, ::-1])
    out_size = get_optimal_out_size(tuple(img.shape[:2]))
    out_img = preprocess(img, out_size)
    return out_img, tuple(img.shape[:2])

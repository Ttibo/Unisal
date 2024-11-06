import os

DATASET_PATHS = {
    # 'SALICON': {
    #     'type' : 'image',
    #     'train': "/Users/coconut/Documents/Dataset/GenSaliency/VisualSaliency/SALICON/",
    #     'val': "/Users/coconut/Documents/Dataset/GenSaliency/VisualSaliency/SALICON/",
    #     'extension' : "jpg"
    # },
    'Packaging_3sec': {
        'source' : 'SALICON',
        'type' : 'image',
        'train': "/Users/coconut/Documents/Dataset/GenSaliency/VisualSaliency/Packaging_delta_3_sigma_20/train/",
        'val': "/Users/coconut/Documents/Dataset/GenSaliency/VisualSaliency/Packaging_delta_3_sigma_20/val/",
        'extension' : "jpg",
        'input_size' : (412,412),
        'target_size' : (512,512)
    },
    # 'Naturel': {
    #     'type' : 'video',
    #     'path': "/Users/coconut/Documents/Dataset/GenSaliency/VisualSaliency/DHF1K/DHF1K/",
    #     'extension' : "jpg",
    #     'img_dir' : "images"
    # },
    # 'UCFSports': {
    #     'type' : 'video',
    #     'path': "/Users/coconut/Documents/Dataset/GenSaliency/VisualSaliency/ucf_sports_actions/",
    #     'extension' : "png",
    #     'img_dir' : "images"
    # },
    # 'DHF1K': {
    #     'type' : 'video',
    #     'path': "/Users/coconut/Documents/Dataset/GenSaliency/VisualSaliency/Ittention_advertising_video/",
    #     'extension' : "jpg",
    #     'img_dir' : "frames"
    # }
}


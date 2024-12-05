import os

DATASET_PATHS = {
    'SALICON': {
        'type' : 'image',
        'train': "/Users/coconut/Documents/Dataset/GenSaliency/VisualSaliency/SALICON/",
        'val': "/Users/coconut/Documents/Dataset/GenSaliency/VisualSaliency/SALICON/",
        'extension' : "jpg"
    },
    'DHF1K': {
        'type' : 'video',
        'path': "/Users/coconut/Documents/Dataset/GenSaliency/VisualSaliency/DHF1K/DHF1K/",
        'extension' : "jpg",
        'img_dir' : "images"
    },
    'UCFSports': {
        'type' : 'video',
        'path': "/Users/coconut/Documents/Dataset/GenSaliency/VisualSaliency/ucf_sports_actions/",
        'extension' : "png",
        'img_dir' : "images"
    },
}


import os

DATASET_PATHS_LOCAL = {
    'SALICON': {
        'type' : 'image',
        'train': "/Users/coconut/Documents/Dataset/GenSaliency/VisualSaliency/SALICON/train/",
        'val': "/Users/coconut/Documents/Dataset/GenSaliency/VisualSaliency/SALICON/val/",
        'extension' : "jpg"
    },
    'Packaging': {
        'type' : 'image',
        'train': "/Users/coconut/Documents/Dataset/GenSaliency/VisualSaliency/Packaging_delta_3_sigma_20/train/",
        'val': "/Users/coconut/Documents/Dataset/GenSaliency/VisualSaliency/Packaging_delta_3_sigma_20/val/",
        'extension' : "jpg"
    },
    'DHF1K': {
        'type' : 'video',
        'train': "/Users/coconut/Documents/Dataset/GenSaliency/VisualSaliency/DHF1K/DHF1K/",
        'val': "/Users/coconut/Documents/Dataset/GenSaliency/VisualSaliency/DHF1K/DHF1K/",
        'extension' : "jpg"
    },
    'UCFSports': {
        'type' : 'video',
        'train': "/Users/coconut/Documents/Dataset/GenSaliency/VisualSaliency/ucf_sports_actions/train/",
        'val': "/Users/coconut/Documents/Dataset/GenSaliency/VisualSaliency/ucf_sports_actions/val/",
        'extension' : "jpg"
    },
    'Advertising': {
        'type' : 'video',
        'train': "/Users/coconut/Documents/Dataset/GenSaliency/VisualSaliency/Ittention_advertising_video/frames/train/",
        'val': "/Users/coconut/Documents/Dataset/GenSaliency/VisualSaliency/Ittention_advertising_video/frames/val/",
        'extension' : "jpg"
    }
}


DATASET_PATHS_SERVER = {
    'SALICON': {
        'type' : 'image',
        'train': "/home/LelongT/Dataset/SALICON/train/",
        'val': "/home/LelongT/Dataset/SALICON/val/",
        'extension' : "jpg"
    },
    'Packaging': {
        'type' : 'image',
        'train': "/home/LelongT/Dataset/Packaging_delta_3_sigma_20/train/",
        'val': "/home/LelongT/Dataset/Packaging_delta_3_sigma_20/val/",
        'extension' : "jpg"
    },
    'DHF1K': {
        'type' : 'video',
        'train': "//home/LelongT/Datasety/DHF1K/",
        'val': "//home/LelongT/Datasety/DHF1K/",
        'extension' : "jpg"
    },
    'UCFSports': {
        'type' : 'video',
        'train': "//home/LelongT/Datasety/ucf_sports_actions/train/",
        'val': "//home/LelongT/Datasety/ucf_sports_actions/val/",
        'extension' : "jpg"
    },
    'Advertising': {
        'type' : 'video',
        'train': "//home/LelongT/Datasety/Ittention_advertising_video/frames/train/",
        'val': "//home/LelongT/Datasety/Ittention_advertising_video/frames/val/",
        'extension' : "jpg"
    }
}



import os

DATASET_PATHS_LOCAL = {
    # 'Packaging': {
    #     'type' : 'image',
    #     'train': "/Users/coconut/Documents/Dataset/GenSaliency/VisualSaliency/SALICON/",
    #     'val': "/Users/coconut/Documents/Dataset/GenSaliency/VisualSaliency/SALICON/",
    #     'extension' : "jpg"
    # },
    'Packaging': {
        'type' : 'image',
        'train': "/Users/coconut/Documents/Dataset/GenSaliency/VisualSaliency/Packaging_delta_3_sigma_20/train/",
        'val': "/Users/coconut/Documents/Dataset/GenSaliency/VisualSaliency/Packaging_delta_3_sigma_20/val/",
        'extension' : "jpg"
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
    # 'Advertising': {
    #     'type' : 'video',
    #     'path': "/Users/coconut/Documents/Dataset/GenSaliency/VisualSaliency/Ittention_advertising_video/",
    #     'extension' : "jpg",
    #     'img_dir' : "frames"
    # }
}


DATASET_PATHS_SERVER = {
    'Packaging': {
        'type' : 'image',
        'train': "/home/LelongT/Dataset/SALICON/",
        'val': "/home/LelongT/Dataset/SALICON/",
        'extension' : "jpg"
    },
    'Packaging': {
        'type' : 'image',
        'train': "/home/LelongT/Dataset/Packaging_delta_1_sigma_20/train/",
        'val': "/home/LelongT/Dataset/Packaging_delta_1_sigma_20/val/",
        'extension' : "jpg"
    },
    'Naturel': {
        'type' : 'video',
        'path': "/home/LelongT/Dataset/DHF1K/",
        'extension' : "jpg",
        'img_dir' : "images"
    },
    'UCFSports': {
        'type' : 'video',
        'path': "/home/LelongT/Dataset/ucf_sports_actions/",
        'extension' : "png",
        'img_dir' : "images"
    },
    'Advertising': {
        'type' : 'video',
        'path': "/home/LelongT/Dataset/ittention_advertising_video/",
        'extension' : "jpg",
        'img_dir' : "frames"
    }
}


DATASET_PATHS_DESKTOP = {
    'Packaging': {
        'type' : 'image',
        'train': "D:/SALICON/",
        'val': "D:/SALICON/",
        'extension' : "jpg"
    },
    'Packaging': {
        'type' : 'image',
        'train': "D:/Packaging_delta_3_sigma_20/train/",
        'val': "D:/Packaging_delta_3_sigma_20/val/",
        'extension' : "jpg"
    },
    'Naturel': {
        'type' : 'video',
        'path': "D:/DHF1K/",
        'extension' : "jpg",
        'img_dir' : "images"

    },
    'UCFSports': {
        'type' : 'video',
        'path': "D:/ucf_sports_actions/",
        'extension' : "png",
        'img_dir' : "images"
    },
    'Advertising': {
        'type' : 'video',
        'path': "D:/Ittention_advertising_video/",
        'extension' : "jpg",
        'img_dir' : "frames"

    }
}




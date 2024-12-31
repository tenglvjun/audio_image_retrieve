#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   main.py
@Time    :   2024/12/31 13:45:25
@Author  :   Tony Teng 
@Version :   1.0
@Contact :   tenglvjun@gmail.com
@Desc    :   None
"""

import os
from PIL import Image
import numpy as np


def read_datasets():
    image_folder = "datasets/image"
    audio_folder = "datasets/audio"
    images = os.listdir(image_folder)
    audios = os.listdir(audio_folder)
    datasets = {}

    for image in images:
        datasets[image.split(".")[0]] = {
            "image": os.path.join(image_folder, image),
        }
    for audio in audios:
        datasets[audio.split(".")[0]]["audio"] = os.path.join(audio_folder, audio)
    return datasets


def load_models():
    import laion_clap
    import open_clip

    clap_model = laion_clap.CLAP_Module(enable_fusion=True, device="cuda")
    clap_model.load_ckpt()
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    clip_model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    return clap_model, clip_model, tokenizer, preprocess


def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def l2_distance(a, b):
    distance = np.linalg.norm(a - b)
    return distance


def main():
    datasets = read_datasets()
    clap_model, clip_model, _, preprocess = load_models()
    audio_embed = clap_model.get_audio_embedding_from_filelist(
        [datasets["bird"]["audio"]]
    )[0]

    for name, dataset in datasets.items():
        image = preprocess(Image.open(dataset["image"])).unsqueeze(0)
        features = clip_model.encode_image(image)
        features /= features.norm(dim=-1, keepdim=True)
        image_embedding = features[0].detach().numpy()
        distance = cosine_distance(image_embedding, audio_embed)
        print(f"{name}: {distance}")


if __name__ == "__main__":
    main()

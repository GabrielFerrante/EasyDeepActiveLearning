import os
import pandas as pd
import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import ast
import random
import pickle
from random import randint

# ORIGINAL DATASET http://ufldl.stanford.edu/housenumbers/

# FUNÇÃO DO CÓDIGO - https://www.kaggle.com/code/shudhanshurp/svhn-cnn-crnn-ctc-loss-file

def mat_to_dataset_grouped(mat_path, image_folder):
    """Convert .mat file to a DataFrame with grouped labels and bounding boxes for each image."""
    df = pd.DataFrame(columns=('filename', 'value', 'digits', 'length', 'width', 'height', 'box'))
    digits = []
    left = top = 999999
    right = bottom = 0
        
    # Normalize the image folder path
    image_folder = os.path.normpath(image_folder)
    print(f"Using normalized image folder path: {image_folder}")
    with h5py.File(mat_path, mode='r') as f:
        # Preload all file names
        file_names = [''.join(chr(c[0]) for c in f[f['digitStruct']['name'][i, 0]][:]) 
                    for i in range(len(f['digitStruct']['name']))]
            
        # Print first few filenames for verification
        print(f"First few filenames in .mat file: {file_names[:5]}")

        # Preload all bounding boxes
        bboxes = []
        for i in range(len(f['digitStruct']['bbox'])):
            box_i = f['digitStruct']['bbox'][i, 0]
            bbox = {key: [] for key in ['label', 'left', 'top', 'width', 'height']}
            for key in bbox.keys():
                attr = f[box_i][key]
                bbox[key] = ([int(f[attr[index, 0]][0, 0]) for index in range(attr.shape[0])] 
                                if attr.shape[0] > 1 else [int(attr[0, 0])])
            bboxes.append(bbox)

        # Process all preloaded data
        for i, (filename, bbox) in enumerate(zip(file_names, bboxes)):
            # Reset bounding box coordinates for each new image
            left = top = 999999
            right = bottom = 0
            digits = []
                
            # Process the digits and bounding boxes
            for digit, l, t, w, h in zip(bbox['label'], bbox['left'], bbox['top'], 
                                        bbox['width'], bbox['height']):
                digit = 0 if digit == 10 else digit
                left = min(l, left)
                top = min(t, top)
                right = max(l + w, right)
                bottom = max(t + h, bottom)
                digits.append(digit)

            # Process the image directly without checking previous state
            img_path = os.path.join(image_folder, filename)
                
            if os.path.isfile(img_path):
                with Image.open(img_path) as img:
                    width, height = img.size
                    df.loc[len(df)] = [
                            filename,
                            int("".join(map(str, digits))),                        digits,
                            len(digits),
                            width,
                            height,
                            [left, top, right, bottom]
                        ]
            else:
                print(f"\nWarning: Could not find image file: {img_path}")

            if (i + 1) % 100 == 0:  # Print progress every 100 images
                print(f'Processing {i + 1}/{len(file_names)} records...', end='\r')

    print(f'\nProcessed {len(file_names)} records.')
    return df

# Defina aqui os caminhos para as suas pastas
diretoriosMat = [
    'F:/SVHN/train/train/digitStruct.mat',  # Substitua pelo caminho real da sua pasta de treino
    'F:/SVHN/test/test/digitStruct.mat',   # Substitua pelo caminho real da sua pasta de teste
    'F:/SVHN/extra/extra/digitStruct.mat'   # Substitua pelo caminho real da sua pasta extra
]

train_df = mat_to_dataset_grouped(diretoriosMat[0], 'F:/SVHN/train/train/')
test_df = mat_to_dataset_grouped(diretoriosMat[1], 'F:/SVHN/test/test/')
extra_df = mat_to_dataset_grouped(diretoriosMat[2], 'F:/SVHN/extra/extra/')

output_dir = '.'
train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
extra_df.to_csv(os.path.join(output_dir, 'extra.csv'), index=False)

print("CSV files saved at:", output_dir)




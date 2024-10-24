import os
import tensorflow as tf
from matplotlib import pyplot as plt
import cv2
import numpy as np
from preprocess import PreProcess_Data  # Assuming preprocess.py contains your PreProcess_Data class
if __name__ == "__main__":
    images_folder_path = 'test'
    imdata = PreProcess_Data()
    imdata.visualization_images(images_folder_path, 4)
    project_df, train, label = imdata.preprocess(images_folder_path)
    class_of_interest = 'positive_class'
    project_df['Binary_Label'] = project_df['Labels'].apply(lambda x: 1 if x == class_of_interest else 0)
    project_df.to_csv("project4_images_binary.csv", index=False)
    print("DataFrame Shape:", project_df.shape)
    print("Number of unique binary labels:", project_df['Binary_Label'].nunique())
    print(project_df['Binary_Label'].value_counts())
#%% 0. SETUP: paths and settings
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
 
# Work from the folder this script is saved in (the repo folder)
try:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass
print("Working directory:", os.getcwd())
 
# Paths (check the dataset folder name matches yours exactly)
DATA_DIR   = os.path.join("Data", "Mars Surface and Curiosity Image")   # dataset folder
IMG_DIR    = os.path.join(DATA_DIR, "images")                           # the pictures
CLASS_FILE = os.path.join(DATA_DIR, "Labels Information.txt")           # number -> class name
OUT_DIR    = os.path.join(DATA_DIR, "processed")                        # results go here
os.makedirs(OUT_DIR, exist_ok=True)
 
IMG_SIZE = 224          # every image gets resized to IMG_SIZE x IMG_SIZE
 
print("Dataset folder contains:", os.listdir(DATA_DIR))
 

#%% 1a. LOAD THE LABEL FILES (builds one table, df, listing every image)
train = pd.read_csv(os.path.join(DATA_DIR, "Train_CSV.csv"))
val   = pd.read_csv(os.path.join(DATA_DIR, "Validation_CSV.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "Test_CSV.csv"))
 
train["split"] = "train"          # remember which split each row came from
val["split"]   = "val"
test["split"]  = "test"
 
df = pd.concat([train, val, test], ignore_index=True)   # stack into one table
df = df.rename(columns={"JPG": "file", "LABELS": "label_id"})
 
# CSV says "calibrated/xxx.JPG", but the images are directly in IMG_DIR,
# so keep only the file name and build the full path from IMG_DIR
df["file"] = df["file"].str.replace("calibrated/", "")
df["path"] = df["file"].apply(lambda name: os.path.join(IMG_DIR, name))
 
# Turn label numbers into class names, e.g. 8 -> "ground"
id_to_name = {}
with open(CLASS_FILE) as f:
    for line in f:
        if line.strip():
            num, name = line.split(maxsplit=1)
            id_to_name[int(num)] = name.strip()
df["class_name"] = df["label_id"].map(id_to_name)
 
print("Total images:", len(df))
print(df["split"].value_counts())
print(df.head())
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
DATA_DIR   = os.path.join("Data", "Mars_Surface_and_Curiosity_Image")   # dataset folder
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

#%% 1b. CHECK FOR MISSING / INCORRECT DATA
n_start = len(df)
 
# (i) labels that don't match any class name
bad_label = df["class_name"].isna()
print("Entries with unknown label:", bad_label.sum())
 
# (ii) image files that are missing or won't open
status = []
for path in df["path"]:
    if not os.path.exists(path):
        status.append("missing")
        continue
    try:
        with Image.open(path) as im:
            im.verify()                   # checks the file is a valid image
        status.append("ok")
    except Exception:
        status.append("corrupted")
df["status"] = status
print(df["status"].value_counts())
 
# (iii) the same image listed more than once
dupes = df.duplicated(subset="file")
print("Duplicate entries:", dupes.sum())
 
# Remove anything bad
df = df[(~bad_label) & (df["status"] == "ok") & (~dupes)].reset_index(drop=True)
print(f"\nREPORT: started with {n_start} entries, removed {n_start - len(df)}, "
      f"{len(df)} remain.")

#%% 1c. STRATIFIED RE-SPLIT (same class proportions in every split)
from sklearn.model_selection import train_test_split

train_df, temp_df = train_test_split(df, test_size=0.30,
                                     stratify=df["class_name"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.50,
                                   stratify=temp_df["class_name"], random_state=42)

df["split"] = "train"
df.loc[val_df.index, "split"] = "val"
df.loc[test_df.index, "split"] = "test"

print(df["split"].value_counts())
print(pd.crosstab(df["class_name"], df["split"]))   # images per class per split

#%% 2a. CATEGORICAL DATA: label encoding
from sklearn.preprocessing import LabelEncoder
 
# Gives each class present a number 0, 1, 2... with no gaps
# (the "sun" class has no images, so 24 classes are used, not 25)
encoder = LabelEncoder()
df["label"] = encoder.fit_transform(df["class_name"])
n_classes = len(encoder.classes_)
 
class_table = pd.DataFrame({"class_name": encoder.classes_,
                            "encoded_label": range(n_classes)})
print("Number of classes:", n_classes)
print(class_table.to_string(index=False))   # copy this table into the report
class_table.to_csv(os.path.join(OUT_DIR, "class_encoding.csv"), index=False)
 
 
#%% 2b. CATEGORICAL DATA: one-hot encoding
one_hot = np.eye(n_classes, dtype=int)[df["label"].values]
print("One-hot shape:", one_hot.shape)
print("Example, first image:", df.loc[0, "class_name"], "->", one_hot[0])
 

#%% 2c. CLASS IMBALANCE (bar chart for the report)
counts = df[df["split"] == "train"]["class_name"].value_counts()
plt.figure(figsize=(10, 5))
counts.plot(kind="bar")
plt.title("Number of training images per class")
plt.ylabel("Images")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "class_counts.png"), dpi=200)
plt.show()
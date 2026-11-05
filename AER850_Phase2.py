#%% 0. SETUP: paths and settings
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageOps
 
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

# Class weights: rare classes get a bigger weight during training
# weight = total training images / (number of classes * images in that class)
from sklearn.utils.class_weight import compute_class_weight
train_labels = df[df["split"] == "train"]["label"].values
present = np.unique(train_labels)
weights = compute_class_weight("balanced", classes=present, y=train_labels)
print("REPORT: largest class:", counts.idxmax(), counts.max(),
      "| smallest class:", counts.idxmin(), counts.min())
print("Class weights:", dict(zip(encoder.inverse_transform(present),
                                 [round(float(w), 2) for w in weights])))

#%% 3a. SCALE AND NORMALIZE: look at the raw images first
sizes, modes = [], []
for path in df["path"].sample(min(300, len(df)), random_state=0):
    with Image.open(path) as im:
        sizes.append(im.size)
        modes.append(im.mode)
print("REPORT: different image sizes in a sample of 300:", len(set(sizes)))
print("Example sizes:", list(set(sizes))[:5])
print("Colour modes:", pd.Series(modes).value_counts().to_dict())  # L = grayscale
 
 
#%% 3b. RESIZE + CONVERT TO RGB, save processed copies
# Saved as PNGs with pixels 0-255. Scaling to 0-1 happens when loading
# (next cell), because saving 6000+ scaled images would use ~4 GB.
PROC_IMG_DIR = os.path.join(OUT_DIR, f"images_{IMG_SIZE}_padded")
os.makedirs(PROC_IMG_DIR, exist_ok=True)
 
new_paths = []
for i, (name, path) in enumerate(zip(df["file"], df["path"])):
    out_path = os.path.join(PROC_IMG_DIR, os.path.splitext(name)[0] + ".png")
    if not os.path.exists(out_path):          # skip images already done
        with Image.open(path) as im:
            # pad to a square with black borders, then resize (keeps proportions)
            ImageOps.pad(im.convert("RGB"), (IMG_SIZE, IMG_SIZE),
                         color=(0, 0, 0)).save(out_path)
    new_paths.append(out_path)
    if i % 500 == 0:
        print(f"{i}/{len(df)} images processed")
df["processed_path"] = new_paths
print("Done resizing.")
 
 
#%% 3c. NORMALIZE: before vs after (figure + numbers for the report)
def load_normalized(path):
    """Load a processed image and scale pixels from 0-255 to 0-1."""
    return np.asarray(Image.open(path), dtype=np.float32) / 255.0
 
raw_arr = np.asarray(Image.open(df.loc[0, "path"]))
norm_arr = load_normalized(df.loc[0, "processed_path"])
 
print("REPORT: BEFORE shape", raw_arr.shape, "| pixel range", raw_arr.min(), "to", raw_arr.max())
print("REPORT: AFTER  shape", norm_arr.shape, "| pixel range",
      round(float(norm_arr.min()), 3), "to", round(float(norm_arr.max()), 3))
 
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
ax[0].imshow(raw_arr, cmap="gray" if raw_arr.ndim == 2 else None)
ax[0].set_title(f"Original {raw_arr.shape[1]}x{raw_arr.shape[0]}, 0-255")
ax[1].imshow(norm_arr)
ax[1].set_title(f"Processed {IMG_SIZE}x{IMG_SIZE}, 0-1")
for a in ax:
    a.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "before_after.png"), dpi=200)
plt.show()
 
# Save the cleaned, encoded image list (part of your "processed data")
df[["file", "class_name", "label", "split"]].to_csv(
    os.path.join(OUT_DIR, "processed_labels.csv"), index=False)

#%% 4a. FEATURE EXTRACTION
# A pretrained ResNet50 (TensorFlow) turns each image into 2048 numbers.

BATCH = 32
paths = df["processed_path"].tolist()
 
try:
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
    model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    feats = []
    for start in range(0, len(paths), BATCH):
        batch = np.stack([np.asarray(Image.open(p), dtype=np.float32)
                          for p in paths[start:start + BATCH]])
        # ResNet50 uses its own normalization, so it gets the 0-255 pixels
        feats.append(model.predict(preprocess_input(batch), verbose=0))
        print(f"{start + len(batch)}/{len(paths)} images")
    features = np.vstack(feats)
    feature_method = "ResNet50 (ImageNet) features"
except ImportError:
    print("TensorFlow not found, using simple 32x32 pixel features instead.")
    features = np.stack([
        np.asarray(Image.open(p).resize((32, 32)), dtype=np.float32).ravel() / 255.0
        for p in paths])
    feature_method = "32x32 pixel features"
 
print("REPORT:", feature_method, "| feature matrix shape:", features.shape)

#%% 4b. FEATURE SELECTION WITH PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
 
train_mask = (df["split"] == "train").values
# Standardize: each feature gets mean 0 and std 1 (fit on training data only)
scaler = StandardScaler().fit(features[train_mask])
features_std = scaler.transform(features)
 
# PCA: find the directions of biggest variation in the training data
pca = PCA().fit(features_std[train_mask])
cum_var = np.cumsum(pca.explained_variance_ratio_)
n_keep = int(np.searchsorted(cum_var, 0.95) + 1)          # components for 95% variance
print(f"REPORT: {features.shape[1]} features reduced to {n_keep} "
      f"principal components (keeps 95% of the variance).")
 
plt.figure(figsize=(7, 4))
plt.plot(np.arange(1, len(cum_var) + 1), cum_var)
plt.axhline(0.95, linestyle="--", color="gray")
plt.axvline(n_keep, linestyle="--", color="gray")
plt.xlabel("Number of principal components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA explained variance")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pca_variance.png"), dpi=200)
plt.show()
 
pca_final = PCA(n_components=n_keep).fit(features_std[train_mask])
reduced = pca_final.transform(features_std)              # keep the top n_keep components
 
 
#%% 4c. 2D VIEW OF THE FEATURES (do similar classes group together?)
pca2 = reduced[:, :2]                                     # first two components
plt.figure(figsize=(9, 7))
top = df["class_name"].value_counts().index[:8]           # 8 biggest classes, readable
for name in top:
    m = (df["class_name"] == name).values
    plt.scatter(pca2[m, 0], pca2[m, 1], s=8, label=name, alpha=0.6)
plt.legend(markerscale=2, fontsize=8)
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title(f"First two principal components ({feature_method})")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pca_2d.png"), dpi=200)
plt.show()
#  preparing datasets , bcz in the downloaded data train test val is not sillted
#  so we spliited the data and save that in a folder under parent dir

import os
import random
import shutil

# ----------------------------------------------------
# Settings - change ARCHIVE if your extracted folder is elsewhere
# ----------------------------------------------------
ARCHIVE = r"C:\Users\Yash\Desktop\pytorch\Garbage_Classifier\archive"
DEST = r"C:\Users\Yash\Desktop\pytorch\Garbage_Classifier\Garbage_Image_Dataset"

SPLITS = {"train": 0.70, "validation": 0.15, "test": 0.15}
EXTS = (".jpg", ".jpeg", ".png")
random.seed(42)


def find_class_root(archive):
    # Walk down until we find the folder that directly contains 'cardboard'
    for root, dirs, _ in os.walk(archive):
        if "cardboard" in [d.lower() for d in dirs]:
            return root
    raise FileNotFoundError("Could not find a folder containing 'cardboard' inside " + archive)


def main():
    source = find_class_root(ARCHIVE)
    print("Using source folder:", source)

    if os.path.exists(DEST):
        raise SystemExit(f"{DEST} already exists. Delete it (or change DEST) and run again, "
                         "otherwise old and new images would be mixed.")

    for class_name in sorted(os.listdir(source)):
        class_dir = os.path.join(source, class_name)
        if not os.path.isdir(class_dir):  # skips the .txt files
            continue

        files = [f for f in os.listdir(class_dir) if f.lower().endswith(EXTS)]
        random.shuffle(files)

        n = len(files)
        n_train = int(n * SPLITS["train"])
        n_val = int(n * SPLITS["validation"])
        parts = {
            "train": files[:n_train],
            "validation": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }

        for split, split_files in parts.items():
            out_dir = os.path.join(DEST, split, class_name)
            os.makedirs(out_dir, exist_ok=True)
            for f in split_files:
                shutil.copy(os.path.join(class_dir, f), os.path.join(out_dir, f))

        print(f"{class_name}: {n} images -> "
              f"{len(parts['train'])}/{len(parts['validation'])}/{len(parts['test'])}")

    print("\nDone. Dataset created at:", DEST)


if __name__ == "__main__":
    main()
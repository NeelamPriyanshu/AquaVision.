print("AquaVision program started!")
import os

# Dataset location
DATASET_PATH = r"D:\AquaVision\Dataset\raw\underwater_trash_detection"

# Dataset folders
splits = ["train", "val", "test"]

print("=" * 50)
print("       AQUAVISION DATASET VERIFICATION")
print("=" * 50)

for split in splits:

    folder_path = os.path.join(DATASET_PATH, split)

    if not os.path.exists(folder_path):
        print(f"\n{split.upper()} folder NOT FOUND!")
        continue

    files = os.listdir(folder_path)

    images = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    txt_files = [f for f in files if f.lower().endswith(".txt")]
    xml_files = [f for f in files if f.lower().endswith(".xml")]

    print(f"\n{split.upper()} DATASET")
    print("-" * 30)
    print("Images:", len(images))
    print("YOLO annotations:", len(txt_files))
    print("XML annotations:", len(xml_files))

print("\nDataset verification completed!")

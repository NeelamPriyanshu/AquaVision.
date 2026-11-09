
import os
import yaml

# Main dataset folder
DATASET_PATH = r"D:\AquaVision\Dataset\raw\underwater_trash_detection"

# Read data.yaml
yaml_path = os.path.join(DATASET_PATH, "data.yaml")

with open(yaml_path, "r") as file:
    data = yaml.safe_load(file)

print("=" * 50)
print("       AQUAVISION YOLO DATASET CHECK")
print("=" * 50)

print("\nNumber of classes:", data["nc"])
print("Class names:", data["names"])

# Check each dataset split
for split in ["train", "val", "test"]:

    folder = os.path.join(DATASET_PATH, data[split])

    if not os.path.isdir(folder):
        print(f"\n{split.upper()}: Folder NOT FOUND!")
        continue

    images = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    missing_labels = 0
    valid_labels = 0
    invalid_labels = 0

    for image in images:
        label_file = os.path.splitext(image)[0] + ".txt"
        label_path = os.path.join(folder, label_file)

        if not os.path.exists(label_path):
            missing_labels += 1
            continue

        try:
            with open(label_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                values = line.strip().split()

                if len(values) != 5:
                    raise ValueError("Incorrect number of values")

                class_id = int(values[0])
                coordinates = list(map(float, values[1:]))

                if class_id < 0 or class_id >= data["nc"]:
                    raise ValueError("Invalid class ID")

                if not all(0 <= x <= 1 for x in coordinates):
                    raise ValueError("Coordinates outside 0-1")

            valid_labels += 1

        except Exception:
            invalid_labels += 1

    print(f"\n{split.upper()} DATASET")
    print("-" * 30)
    print("Images:", len(images))
    print("Valid annotation files:", valid_labels)
    print("Missing annotation files:", missing_labels)
    print("Invalid annotation files:", invalid_labels)

print("\nDataset checking completed!")

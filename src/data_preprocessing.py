
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


def resize_images(
    source_folder,
    output_folder,
    image_size=(224, 224),
    jpeg_quality=80
):
    """Resize images stored in class-wise folders."""

    source_folder = Path(source_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    supported_formats = {".jpg", ".jpeg", ".png", ".bmp"}

    for class_folder in source_folder.iterdir():
        if not class_folder.is_dir():
            continue

        output_class = output_folder / class_folder.name
        output_class.mkdir(parents=True, exist_ok=True)

        for image_path in class_folder.iterdir():
            if image_path.suffix.lower() not in supported_formats:
                continue

            output_path = output_class / f"{image_path.stem}.jpg"

            if output_path.exists():
                continue

            try:
                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                    image = image.resize(image_size)
                    image.save(
                        output_path,
                        format="JPEG",
                        quality=jpeg_quality
                    )
            except Exception as error:
                print(f"Could not process {image_path}: {error}")


def create_manifest(resized_folder, manifest_file):
    """Create a CSV containing image paths and class labels."""

    resized_folder = Path(resized_folder)
    manifest_file = Path(manifest_file)

    records = []

    for class_folder in sorted(resized_folder.iterdir()):
        if not class_folder.is_dir():
            continue

        for image_path in sorted(class_folder.glob("*.jpg")):
            records.append({
                "image_path": str(image_path),
                "label": class_folder.name
            })

    manifest = pd.DataFrame(records, columns=["image_path", "label"])

    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_file, index=False)

    return manifest


def split_manifest(data, results_folder, random_seed=42):
    """Split the dataset into 70% train, 20% validation and 10% test."""

    results_folder = Path(results_folder)
    results_folder.mkdir(parents=True, exist_ok=True)

    train_df, temp_df = train_test_split(
        data,
        test_size=0.30,
        random_state=random_seed,
        stratify=data["label"]
    )

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=1 / 3,
        random_state=random_seed,
        stratify=temp_df["label"]
    )

    train_df.to_csv(results_folder / "train_split.csv", index=False)
    validation_df.to_csv(
        results_folder / "validation_split.csv", index=False
    )
    test_df.to_csv(results_folder / "test_split.csv", index=False)

    return train_df, validation_df, test_df

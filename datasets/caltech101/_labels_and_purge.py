if __name__ == "__main__":
    import pathlib
    import shutil

    base_path = pathlib.Path("datasets", "caltech101")
    images_path = base_path / "images"
    labels_path = base_path / "labels.txt"

    NEW_CNAMES = {
        "airplanes": "airplane",
        "Faces": "face",
        "Leopards": "leopard",
        "Motorbikes": "motorbike",
    }

    # Ignore and delete same classes as MVPR
    for class_ in ["BACKGROUND_Google", "Faces_easy"]:
        class_path = images_path / class_
        if class_path.exists():
            shutil.rmtree(images_path / class_)

    with open(labels_path, mode="w", encoding="utf-8") as labels_file:
        for class_path in images_path.iterdir():
            current_label = class_path.name
            display_label = NEW_CNAMES.get(
                current_label, current_label.lower())
            for image_path in class_path.iterdir():
                display_path = "/".join(image_path.parts[-2:])
                labels_file.write(f"{display_path}  {display_label}\n")

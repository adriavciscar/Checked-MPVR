if __name__ == "__main__":
    import pathlib

    base_path = pathlib.Path("datasets", "cubs")
    images_meta_path = base_path / "images.txt"
    test_path = base_path / "train_test_split.txt"
    images_path = base_path / "images"
    labels_path = base_path / "labels.txt"

    id_paths: dict[str, str] = {}
    with open(images_meta_path, mode="r", encoding="utf-8") as meta_file:
        for line in meta_file:
            id_, image_path = line.strip().split(" ")
            id_paths[id_] = image_path

    # Keep only test images
    with open(test_path, mode="r", encoding="utf-8") as test_file:
        for line in test_file:
            id_, is_train = line.strip().split(" ")
            if is_train == "0":
                continue
            train_image_path = images_path / id_paths[id_]
            train_image_path.unlink()

    with open(labels_path, mode="w", encoding="utf-8") as labels_file:
        for class_path in images_path.iterdir():
            current_label = class_path.name
            display_label = current_label[4:].replace("_", " ").lower()
            for image_path in class_path.iterdir():
                display_path = "/".join(image_path.parts[-2:])
                labels_file.write(f"{display_path}  {display_label}\n")

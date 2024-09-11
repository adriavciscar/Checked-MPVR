if __name__ == "__main__":
    import pathlib

    base_path = pathlib.Path("datasets", "in-r")
    meta_path = base_path / "README.txt"
    images_path = base_path / "images"
    labels_path = base_path / "labels.txt"

    labels_cname: dict[str, str] = {}
    with open(meta_path, mode="r", encoding="utf-8") as test_file:
        for line in test_file:
            if not line.startswith("n"):
                continue
            class_dir, label = line.strip().split(" ")
            label = label.replace("_", " ").lower()
            labels_cname[class_dir] = label

    with open(labels_path, mode="w", encoding="utf-8") as labels_file:
        for class_path in images_path.iterdir():
            current_label = class_path.name
            display_label = labels_cname[current_label]
            for image_path in class_path.iterdir():
                display_path = "/".join(image_path.parts[-2:])
                labels_file.write(f"{display_path}  {display_label}\n")

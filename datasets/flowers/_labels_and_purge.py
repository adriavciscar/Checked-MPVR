if __name__ == "__main__":
    import pathlib

    base_path = pathlib.Path("datasets", "flowers")
    meta_path = base_path / "oxford_flower_102_name.csv"
    test_path = base_path / "test.csv"
    classes_path = base_path / "labels.csv"
    images_path = base_path / "images"
    labels_path = base_path / "labels.txt"

    label_cnames: dict[int, str] = {}
    with open(meta_path, mode="r", encoding="utf-8") as meta_file:
        meta_rows = meta_file.readlines()
        for line in meta_rows[1:]:
            label, display = line.strip().split(",")
            label = int(label) + 1  # MATLAB convention
            label_cnames[label] = display

    with open(classes_path, mode="r", encoding="utf-8") as classes_file:
        all_labels = classes_file.read().strip().split(",")

    with open(test_path, mode="r", encoding="utf-8") as test_file:
        test_images = {int(image_idx) - 1  # MATLAB convention
                       for image_idx in test_file.read().strip().split(",")}

    def _get_image_idx(name: str) -> int:
        ini = name.find("_") + 1
        return int(name[ini:ini + 5])

    with open(labels_path, mode="w", encoding="utf-8") as labels_file:
        for image_path in images_path.iterdir():
            display_path = image_path.name
            image_idx = _get_image_idx(display_path)
            if image_idx in test_images:
                current_label = all_labels[image_idx]
                display_label = label_cnames[int(current_label)]
                labels_file.write(f"{display_path}  {display_label}\n")
            else:
                image_path.unlink()

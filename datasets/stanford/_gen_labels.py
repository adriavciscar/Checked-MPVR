if __name__ == "__main__":
    import pathlib
    import json

    base_path = pathlib.Path("datasets", "stanford")
    meta_path = base_path / "class_names.csv"
    classes_path = base_path / "annotations.json"
    images_path = base_path / "images"
    labels_path = base_path / "labels.txt"

    with open(meta_path, mode="r", encoding="utf-8") as meta_file:
        all_labels = meta_file.read().strip().split(",")

    with open(classes_path, mode="r", encoding="utf-8") as classes_file:
        annot = json.load(classes_file)

    with open(labels_path, mode="w", encoding="utf-8") as labels_file:
        for image_dict in annot:
            display_path = image_dict["fname"]
            current_label = int(image_dict["class"])
            display_label = all_labels[current_label - 1]
            labels_file.write(f"{display_path}  {display_label}\n")

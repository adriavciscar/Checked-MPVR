if __name__ == "__main__":
    import pathlib

    base_path = pathlib.Path("datasets", "food101")
    test_path = base_path / "test.txt"
    images_path = base_path / "images"
    labels_path = base_path / "labels.txt"

    with open(test_path, mode="r", encoding="utf-8") as test_file:
        test_images = {f"{line.strip()}.jpg" for line in test_file.readlines()}

    with open(labels_path, mode="w", encoding="utf-8") as labels_file:
        for class_path in images_path.iterdir():
            current_label = class_path.name
            display_label = current_label.replace("_", " ")
            for image_path in class_path.iterdir():
                display_path = "/".join(image_path.parts[-2:])
                if display_path in test_images:
                    labels_file.write(f"{display_path}  {display_label}\n")
                else:
                    image_path.unlink()

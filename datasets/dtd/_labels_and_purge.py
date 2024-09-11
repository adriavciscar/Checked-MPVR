if __name__ == "__main__":
    import pathlib

    base_path = pathlib.Path("datasets", "dtd")
    test_path = base_path / "test1.txt"
    images_path = base_path / "images"
    labels_path = base_path / "labels.txt"

    # Keep only test images
    with open(test_path, mode="r", encoding="utf-8") as test_file:
        test_images = set(test_file.read().splitlines())

    with open(labels_path, mode="w", encoding="utf-8") as labels_file:
        for class_path in images_path.iterdir():
            display_label = class_path.name
            for image_path in class_path.iterdir():
                display_path = "/".join(image_path.parts[-2:])
                if display_path in test_images:
                    labels_file.write(f"{display_path}  {display_label}\n")
                else:
                    image_path.unlink()

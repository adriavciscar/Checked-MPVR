if __name__ == "__main__":
    import pathlib

    base_path = pathlib.Path("datasets", "pets")
    test_path = base_path / "test.txt"
    images_path = base_path / "images"
    labels_path = base_path / "labels.txt"

    test_images: set[str] = set()
    with open(test_path, mode="r", encoding="utf-8") as test_file:
        for line in test_file:
            image_name, *_ = line.strip().split(" ")
            test_images.add(f"{image_name}.jpg")

    with open(labels_path, mode="w", encoding="utf-8") as labels_file:
        for image_path in images_path.iterdir():
            display_path = image_path.name
            display_label = " ".join(display_path[:-4].split("_")[:-1]).lower()
            if display_path in test_images:
                labels_file.write(f"{display_path}  {display_label}\n")
            else:
                image_path.unlink()

if __name__ == "__main__":
    import pathlib

    base_path = pathlib.Path("datasets", "eurosat")
    test_path = base_path / "test.csv"
    images_path = base_path / "images"
    labels_path = base_path / "labels.txt"

    NEW_CNAMES = {
        "AnnualCrop": "annual crop land",
        "Forest": "forest",
        "HerbaceousVegetation": "herbaceous vegetation land",
        "Highway": "highway or road",
        "Industrial": "industrial buildings",
        "Pasture": "pasture land",
        "PermanentCrop": "permanent crop land",
        "Residential": "residential buildings",
        "River": "river",
        "SeaLake": "sea or lake",
    }

    test_images: dict[str, str] = {}
    with open(test_path, mode="r", encoding="utf-8") as test_file:
        test_rows = test_file.readlines()
        for row in test_rows[1:]:
            _, image_path, _, label = row.strip().split(",")
            test_images[image_path] = label

    with open(labels_path, mode="w", encoding="utf-8") as labels_file:
        for class_path in images_path.iterdir():
            current_label = class_path.name
            display_label = NEW_CNAMES.get(
                current_label, current_label.lower())
            for image_path in class_path.iterdir():
                display_path = "/".join(image_path.parts[-2:])
                if display_path in test_images:
                    labels_file.write(f"{display_path}  {display_label}\n")
                else:
                    image_path.unlink()

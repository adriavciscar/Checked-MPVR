"""Eliminate images not in the test set and fix the labels file."""


if __name__ == "__main__":
    import pathlib

    from tqdm import tqdm

    base_path = pathlib.Path("datasets", "aircraft")
    images_path = base_path / "images"
    og_labels_path = base_path / "images_family_test.txt"
    labels_path = base_path / "labels.txt"

    image_files: set[str] = set()
    with open(og_labels_path, mode="r", encoding="utf-8") as og_labels_file:
        labels_lines = og_labels_file.readlines()

    with open(labels_path, mode="w", encoding="utf-8") as labels_file:
        for line in tqdm(labels_lines, desc="Labels fixing"):
            file, *label = line.split(" ")
            file = f"{file}.jpg"
            label = " ".join(label)
            image_files.add(file)
            labels_file.write(f"{file}  {label}")

    for root, _, files in images_path.walk():
        for name in tqdm(files, desc="Image purge"):
            if name not in image_files:
                (root / name).unlink()

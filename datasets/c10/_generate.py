"""Generate the CIFAR10 dataset."""
if __name__ == "__main__":
    import pickle
    import pathlib

    import numpy as np
    import numpy.typing as npt
    import PIL.Image

    from tqdm import tqdm

    base_path = pathlib.Path("datasets", "c10")

    meta_path = base_path / "batches.meta"
    batch_path = base_path / "test_batch"
    images_path = base_path / "images"
    labels_file = base_path / "labels.txt"

    with open(batch_path, mode="rb") as fo:
        image_dict = pickle.load(fo, encoding='bytes')

    images: npt.NDArray[np.uint8] = image_dict[b"data"]
    labels: list[int] = image_dict[b"labels"]

    with open(meta_path, mode="rb") as fo:
        labels_dict = pickle.load(fo, encoding="bytes")

    label_names: list[str] = [name.decode("utf-8")
                              for name in labels_dict[b"label_names"]]

    with open(labels_file, mode="w", encoding="utf-8") as labels_file:
        for idx, (image, label) in enumerate(zip(tqdm(images, desc="Generating"),
                                                 labels)):
            image_path = images_path/f"{idx + 1:05}.jpg"
            # Following the order of the array to get an image
            img_shaped_arr = np.transpose(
                np.reshape(image, (3, 32, 32)), (1, 2, 0))
            pil_img = PIL.Image.fromarray(img_shaped_arr, mode="RGB")
            pil_img.save(image_path)
            pil_img.close()
            labels_file.write(f"{idx + 1:05}.jpg  {label_names[label]}\n")

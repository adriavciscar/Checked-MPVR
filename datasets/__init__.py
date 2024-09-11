"""Module of datasets util."""
import pathlib


def get_dataset_files(ds_nick: str) -> tuple[list[str], list[str]]:
    """Get the filenames and labels of a dataset"""
    files: list[str] = []
    labels: list[str] = []
    ds_labels_path = pathlib.Path("datasets", ds_nick, "labels.txt")
    with open(ds_labels_path, mode="r", encoding="utf-8") as labels_file:
        for line in labels_file:
            file, label = line.split("  ")
            files.append(file)
            labels.append(label)
    return files, labels

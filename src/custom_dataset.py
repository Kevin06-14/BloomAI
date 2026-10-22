import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class FlowersCsvDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.transform = transform
        with Path(csv_path).open("r", encoding="utf-8", newline="") as file:
            self.rows = list(csv.DictReader(file))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"])

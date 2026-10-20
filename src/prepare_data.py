from pathlib import Path

from torchvision import datasets


def main():
    data_dir = Path("data")
    print("Preparing Flowers102 dataset in:", data_dir)
    datasets.Flowers102(
        root=data_dir,
        split="train",
        download=True,
    )
    datasets.Flowers102(
        root=data_dir,
        split="test",
        download=True,
    )
    print("Flowers102 dataset is ready.")


if __name__ == "__main__":
    main()

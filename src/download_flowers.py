from pathlib import Path

from torchvision import datasets


def main():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    dataset = datasets.Flowers102(
        root=data_dir,
        split="train",
        download=True,
    )

    print("Dataset downloaded successfully")
    print("Number of images:", len(dataset))
    print("Number of classes:", 102)
    print("First label:", dataset[0][1])


if __name__ == "__main__":
    main()
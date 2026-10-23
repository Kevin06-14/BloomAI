import tarfile
import urllib.request
from pathlib import Path


URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"


def main():
    data_dir = Path("data")
    archive_path = data_dir / "flower_photos.tgz"
    flower_dir = data_dir / "flower_photos"
    data_dir.mkdir(parents=True, exist_ok=True)

    if not flower_dir.exists():
        if not archive_path.exists():
            print("Downloading TensorFlow Flowers dataset...")
            urllib.request.urlretrieve(URL, archive_path)
        print("Extracting dataset...")
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(data_dir)

    print("TensorFlow Flowers dataset is ready:", flower_dir)


if __name__ == "__main__":
    main()

import os
import urllib.request
import zipfile

# WARNING: This dataset is large (40GB).

def download_and_extract(url, output_dir):
    zip_path = os.path.join(output_dir, os.path.basename(url))
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

output_dir = "./data/coco"
os.makedirs(output_dir, exist_ok=True)

download_and_extract("http://images.cocodataset.org/zips/train2017.zip", output_dir)
download_and_extract("http://images.cocodataset.org/zips/val2017.zip", output_dir)
download_and_extract("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", output_dir)

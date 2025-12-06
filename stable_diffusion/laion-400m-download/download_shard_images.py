import os
import glob
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image
from io import BytesIO
import logging
import mimetypes

# Configuration
SHARDS_META_DIR = "/home/doshlom4/work/laion-400m/shards-meta"
MAX_SHARDS_IN_PARALLEL = 16  # Number of shards to process at once (controls RAM usage)
NUM_THREADS_PER_SHARD = 64  # Number of threads per shard for image downloads
TIMEOUT = 10 # Timeout for image requests in seconds


def setup_shard_logger(log_path):
    logger = logging.getLogger(log_path)
    logger.setLevel(logging.INFO)
    # Remove all handlers
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def is_valid_image(content):
    try:
        img = Image.open(BytesIO(content))
        img.verify()
        return True
    except Exception:
        return False


def download_image(row, images_dir, timeout=TIMEOUT):
    url = row["URL"]
    sample_id = row.name  # Use DataFrame index as ID

    try:
        # Get the image from the URL
        response = requests.get(url, timeout=timeout)
        if response.status_code != 200:
            return sample_id, False, f"HTTP {response.status_code}"
        
        # Check if the content type is an image
        content_type = response.headers.get("Content-Type", "")
        if not (content_type.startswith("image/")):
            return sample_id, False, f"Not an image: {content_type}"
        
        # Determine the file extension
        ext = mimetypes.guess_extension(content_type.split(';')[0]) or ".jpg"
        image_path = os.path.join(images_dir, f"image_{sample_id}{ext}")

        # Check if the image already exists, don't download again
        if os.path.exists(image_path):
            return sample_id, True, "Already exists"
        
        # Check if the image is valid before saving
        if not is_valid_image(response.content):
            return sample_id, False, "Invalid image file"
        
        # Save the image
        with open(image_path, "wb") as f:
            f.write(response.content)
        
        return sample_id, True, None
    except Exception as e:
        return sample_id, False, str(e)


def process_shard(shard_path):
    shard_name = os.path.splitext(os.path.basename(shard_path))[0]
    shard_dir = os.path.dirname(shard_path)
    log_path = os.path.join(shard_dir, f"log_download_shard_{shard_name}.log")
    logger = setup_shard_logger(log_path)

    logger.info(f"Processing shard: {shard_name}")
    images_dir = os.path.join(shard_dir, f"images_{shard_name.split('_')[-1]}")
    os.makedirs(images_dir, exist_ok=True)
    logger.info(f"Images will be saved to: {images_dir}")

    logger.info(f"Reading shard: {shard_path}")
    try:
        df = pd.read_parquet(shard_path, columns=["URL"])
    except Exception as e:
        logger.error(f"Failed to read parquet: {e}")
        return
    total = len(df)
    logger.info(f"Total samples: {total}")
    success, failed = 0, 0
    failed_ids = []
    with ThreadPoolExecutor(max_workers=NUM_THREADS_PER_SHARD) as executor:
        futures = {executor.submit(download_image, row, images_dir): idx for idx, row in df.iterrows()}
        for future in as_completed(futures):
            sample_id, ok, reason = future.result()
            if ok:
                success += 1
                logger.info(f"OK: {sample_id}, Reason: {reason}")
            else:
                failed += 1
                failed_ids.append({"id": int(sample_id), "reason": reason})
                logger.warning(f"Failed: {sample_id} Reason: {reason}")
    logger.info(f"Download complete. Success: {success}, Failed: {failed}")
    logger.info(f"Failed IDs: {failed_ids}")


def main():
    shard_dirs = sorted([d for d in glob.glob(os.path.join(SHARDS_META_DIR, "*")) if os.path.isdir(d)])
    all_shards = []
    for shard_dir in shard_dirs:
        all_shards.extend(sorted(glob.glob(os.path.join(shard_dir, "shard_*.parquet"))))
    print(f"Found {len(all_shards)} shard files.")

    # Process shards in batches to control RAM usage
    for i in range(0, len(all_shards), MAX_SHARDS_IN_PARALLEL):
        batch = all_shards[i:i+MAX_SHARDS_IN_PARALLEL]
        procs = []
        for shard_path in batch:
            p = os.fork()
            if p == 0:
                process_shard(shard_path)
                os._exit(0)
            else:
                procs.append(p)
        for p in procs:
            os.waitpid(p, 0)

if __name__ == "__main__":
    main()

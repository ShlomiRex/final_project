import pandas as pd
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Process
from pathlib import Path
from PIL import Image
from io import BytesIO
import glob
import json
import logging
import sys
import re

def setup_process_logger(save_path):
    pid = os.getpid()
    logger = logging.getLogger()
    for h in logger.handlers[:]:  # Remove existing handlers
        logger.removeHandler(h)
    
    log_file = f"log_{pid}.log"
    if save_path is not None:
        # Save logs to the specified directory
        log_path = os.path.join(save_path, log_file)
        handlers=[
            logging.FileHandler(log_path)
        ]
    else:
        # Empty save_path - log to stdout only
        log_path = log_file
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] [PID %(process)d] %(message)s',
        handlers = handlers
    )
    return logging.getLogger(__name__)

# Configuration
PARQUET_DIR = "/RG/rg-miray/doshlom4/laion-400m/laion400m-meta"
SHARDS_OUT_DIR = "/RG/rg-miray/doshlom4/laion-400m/shards-meta"
SHARD_SIZE = 30_000  # Number of rows per shard

def process_parquet_file(parquet_file):
    # Convert `part-00000-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet` to `00000` and so on
    match = re.search(r"part-(\d{5})-", parquet_file)
    if match:
        parquet_part_num = match.group(1)
    else:
        logger.error(f"Could not extract part number from {parquet_file}")
        raise ValueError(f"Invalid parquet file name: {parquet_file}")

    # Shard directory. Example: `/RG/rg-miray/doshlom4/laion-400m/shards-meta/00000`, `/RG/rg-miray/doshlom4/laion-400m/shards-meta/00001`, etc.
    shard_path = os.path.join(SHARDS_OUT_DIR, parquet_part_num)
    os.makedirs(shard_path, exist_ok=True)

    logger = setup_process_logger(shard_path) # Save logs to each shard directory

    # Read the big parquet file
    logger.info(f"Processing {parquet_file}")
    df = pd.read_parquet(parquet_file)
    logger.info(f"Loaded {len(df)} rows from {parquet_file}")

    # Ensure required columns are present
    assert "URL" in df.columns and "TEXT" in df.columns, "Missing 'URL' or 'TEXT' columns"
    before_filtering = len(df)
    df = df.dropna(subset=["URL", 'TEXT']) # Remove rows with missing URL or TEXT
    after_filtering = len(df)
    logger.info(f"Filtering missing URLs and TEXT: {before_filtering} -> {after_filtering} rows ({after_filtering - before_filtering} removed)")

    total_rows = len(df)
    num_shards = (total_rows + SHARD_SIZE - 1) // SHARD_SIZE
    logger.info(f"Total rows: {total_rows}, Number of shards: {num_shards}")

    for shard_id in range(num_shards):
        # Slice the DataFrame for the current shard
        logger.info(f"Processing shard {shard_id + 1}/{num_shards}")
        shard_df = df.iloc[shard_id*SHARD_SIZE : (shard_id+1)*SHARD_SIZE]
        
        # Save the shard DataFrame to a parquet file
        shard_output_path = os.path.join(shard_path, f"shard_{shard_id:05d}.parquet")
        shard_df.to_parquet(shard_output_path, index=True)
        logger.info(f"Shard {shard_id + 1}/{num_shards} saved to {shard_output_path}")


def main():
    logger = setup_process_logger(save_path=None)
    logger.info("Starting sharding process")
    parquet_files = sorted(glob.glob(os.path.join(PARQUET_DIR, "*.parquet")))
    logger.info(f"Found {len(parquet_files)} parquet files to process.")

    max_procs = 4
    running = []
    for i, parquet_file in enumerate(parquet_files):
        # Start a new process for each parquet file, but limit to max_procs at a time
        while len(running) >= max_procs:
            for p in running:
                if not p.is_alive():
                    logger.info(f"Process {p.pid} finished.")
                    p.join()
            running = [p for p in running if p.is_alive()]
        p = Process(target=process_parquet_file, args=(parquet_file,))
        p.start()
        running.append(p)
    # Wait for all remaining processes to finish
    for p in running:
        logger.info(f"Waiting for process {p.pid} to finish.")
        p.join()
    logger.info("All processes completed successfully.")

if __name__ == "__main__":
    main()
"""
merge_datasets.py - Merge multiple dataset contributions into a single Master Dataset.

Usage:
    python src/data_collection/merge_datasets.py --input data/contributions --output dataset_master

Structure expected in --input:
    data/contributions/
    ├── Member1/
    │   └── dataset/
    │       ├── metadata.csv
    │       ├── raw_eyes/
    │       ├── train/
    │       └── test/
    ├── Member2/
    │   └── dataset/
    │       ...
"""

import os
import sys
import csv
import glob
import shutil
import argparse

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    parser = argparse.ArgumentParser(description="Merge multiple dataset folders into one Master Dataset.")
    parser.add_argument("--input", type=str, default="data/contributions", help="Directory containing team members' folders")
    parser.add_argument("--output", type=str, default="dataset_master", help="Output directory for the master dataset")
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output

    if not os.path.isdir(input_dir):
        print(f"[ERROR] Input directory '{input_dir}' does not exist.")
        print("Please create it and place team members' dataset folders inside.")
        sys.exit(1)

    # Setup output structure
    ensure_dir(output_dir)
    ensure_dir(os.path.join(output_dir, "raw_eyes"))
    for split in ["train", "test"]:
        for label in ["open", "closed"]:
            ensure_dir(os.path.join(output_dir, split, label))

    master_csv_path = os.path.join(output_dir, "metadata_master.csv")
    csv_headers = None
    all_records = []

    total_images_copied = 0
    members_processed = 0

    print("=" * 60)
    print("  DATASET MERGE TOOL")
    print("=" * 60)

    # Find all metadata.csv files in the subdirectories
    search_pattern = os.path.join(input_dir, "**", "metadata.csv")
    csv_files = glob.glob(search_pattern, recursive=True)

    if not csv_files:
        print(f"[WARNING] No metadata.csv found in {input_dir} or its subdirectories.")
        print("Make sure you extract team members' dataset.zip properly.")
        sys.exit(1)

    for csv_file in csv_files:
        dataset_root = os.path.dirname(csv_file)
        
        # Extract the member folder name directly under input_dir
        rel_path = os.path.relpath(csv_file, input_dir)
        member_name = rel_path.split(os.sep)[0]
        if member_name == "" or member_name == ".":
            member_name = "Unknown"

        print(f"Processing contribution from: {member_name} ({dataset_root})")
        members_processed += 1

        # Read CSV
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if csv_headers is None:
                csv_headers = reader.fieldnames
                if "contributor" not in csv_headers:
                    csv_headers.append("contributor")

            for row in reader:
                row["contributor"] = member_name
                
                # Copy the image file if it exists and status is success
                if row.get("status") == "success" and row.get("image_path"):
                    # The image_path in CSV is relative to the dataset root, 
                    # but in our script it's relative to the execution dir or absolute.
                    # Let's reconstruct the path based on dataset_root
                    
                    # original image path might be 'dataset/raw_eyes/...', we just need the basename
                    img_basename = os.path.basename(row["image_path"])
                    
                    # If dataset_root already ends with raw_eyes, the image is directly in dataset_root
                    if os.path.basename(dataset_root) == "raw_eyes":
                        src_raw_img = os.path.join(dataset_root, img_basename)
                        dataset_parent = os.path.dirname(dataset_root)
                    else:
                        src_raw_img = os.path.join(dataset_root, "raw_eyes", img_basename)
                        dataset_parent = dataset_root
                    
                    # Copy to master raw_eyes
                    dst_raw_img = os.path.join(output_dir, "raw_eyes", img_basename)
                    
                    if os.path.exists(src_raw_img):
                        if not os.path.exists(dst_raw_img): # Avoid copying same file twice if somehow duplicated
                            shutil.copy2(src_raw_img, dst_raw_img)
                            total_images_copied += 1
                        row["image_path"] = os.path.normpath(os.path.join(output_dir, "raw_eyes", img_basename)).replace('\\', '/')
                    
                    # Copy train/test splits if they exist
                    final_label = row.get("final_label", "")
                    if final_label in ["open", "closed"]:
                        # Check train
                        src_train = os.path.join(dataset_parent, "train", final_label, img_basename)
                        if os.path.exists(src_train):
                            shutil.copy2(src_train, os.path.join(output_dir, "train", final_label, img_basename))
                        
                        # Check test
                        src_test = os.path.join(dataset_parent, "test", final_label, img_basename)
                        if os.path.exists(src_test):
                            shutil.copy2(src_test, os.path.join(output_dir, "test", final_label, img_basename))

                all_records.append(row)

    # Write Master CSV
    if all_records:
        with open(master_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writeheader()
            writer.writerows(all_records)

    print("=" * 60)
    print(f"[DONE] Merge complete!")
    print(f"  Contributions merged: {members_processed}")
    print(f"  Total records in CSV: {len(all_records)}")
    print(f"  Total unique raw images copied: {total_images_copied}")
    print(f"  Master Dataset Path : {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()

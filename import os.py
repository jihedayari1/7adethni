import os
import gzip
import shutil

root_dir = r"C:\Users\jihed\OneDrive\Desktop\7adethni"  # Change this

gz_files = []

# Find all .gz files
for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.lower().endswith(".gz"):
            gz_files.append(os.path.join(root, file))

print(f"Found {len(gz_files)} .gz files:\n")

for f in gz_files:
    print(f)

print("\nStarting extraction...\n")

for gz_path in gz_files:
    try:
        output_path = os.path.splitext(gz_path)[0]

        with gzip.open(gz_path, "rb") as f_in:
            with open(output_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        print(f"[OK] Extracted: {gz_path}")

    except Exception as e:
        print(f"[ERROR] {gz_path}: {e}")

print("\nFinished.")
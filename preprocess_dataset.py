import os
import json
import numpy as np
from PIL import Image
import base64
import zlib
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import shutil
from io import BytesIO

# --- CONFIGURATION ---

# The root directory of your current, un-processed dataset.
# This should point to the folder containing 'train', 'test', and 'meta.json'.
INPUT_DATA_ROOT = 'datasets/FoodSeg103'

# The root directory where the newly formatted dataset will be saved.
OUTPUT_DATA_ROOT = 'datasets/FoodSeg103_processed'

# This class list is taken directly from swin_tuna/mmseg/datasets/foodseg103.py
# to ensure the class IDs match perfectly.
CLASS_NAMES = [
    "background", "candy", "egg tart", "french fries", "chocolate", "biscuit",
    "popcorn", "pudding", "ice cream", "cheese butter", "cake", "wine",
    "milkshake", "coffee", "juice", "milk", "tea", "almond", "red beans",
    "cashew", "dried cranberries", "soy", "walnut", "peanut", "egg", "apple",
    "date", "apricot", "avocado", "banana", "strawberry", "cherry",
    "blueberry", "raspberry", "mango", "olives", "peach", "lemon", "pear",
    "fig", "pineapple", "grape", "kiwi", "melon", "orange", "watermelon",
    "steak", "pork", "chicken duck", "sausage", "fried meat", "lamb",
    "sauce", "crab", "fish", "shellfish", "shrimp", "soup", "bread", "corn",
    "hamburg", "pizza", "hanamaki baozi", "wonton dumplings", "pasta",
    "noodles", "rice", "pie", "tofu", "eggplant", "potato", "garlic",
    "cauliflower", "tomato", "kelp", "seaweed", "spring onion", "rape",
    "ginger", "okra", "lettuce", "pumpkin", "cucumber", "white radish",
    "carrot", "asparagus", "bamboo shoots", "broccoli", "celery stick",
    "cilantro mint", "snow peas", "cabbage", "bean sprouts", "onion",
    "pepper", "green beans", "French beans", "king oyster mushroom",
    "shiitake", "enoki mushroom", "oyster mushroom", "white button mushroom",
    "salad", "other ingredients"
]

# --- SCRIPT LOGIC ---

def create_class_maps(meta_json_path):
    """
    Creates two essential mappings from the meta.json file:
    1. A map from the original `classId` in annotations to the `classTitle`.
    2. A map from the `classTitle` to our new, sequential integer ID (0-103).
    """
    with open(meta_json_path, 'r') as f:
        meta_data = json.load(f)
    
    original_id_to_title = {cls['id']: cls['title'] for cls in meta_data['classes']}
    title_to_new_id = {class_name: i for i, class_name in enumerate(CLASS_NAMES)}
    
    return original_id_to_title, title_to_new_id

def base64_to_mask(base64_string):
    """
    Decodes a base64 string, which is zlib compressed, into a NumPy array.
    This is the corrected and robust decoding function.
    """
    # 1. Decode the base64 string
    encoded_data = base64.b64decode(base64_string)
    # 2. Decompress the zlib data
    decompressed_data = zlib.decompress(encoded_data)
    # 3. The result is raw bytes of a PNG file. We read it in memory.
    img = Image.open(BytesIO(decompressed_data))
    # 4. Convert the image to a NumPy array. The values will be 0 or 255.
    return np.array(img)

def process_file(args):
    """
    Processes a single annotation JSON file.
    - Reads the JSON.
    - Creates a PNG mask.
    - Copies the corresponding JPG image.
    """
    json_path, original_id_to_title, title_to_new_id, split, process_id = args
    
    try:
        base_filename = os.path.basename(json_path).replace('.jpg.json', '')
        output_mask_path = os.path.join(OUTPUT_DATA_ROOT, 'Images', 'ann_dir', split, f"{base_filename}.png")
        
        # --- RESUMABILITY CHECK ---
        # If the mask already exists, skip processing this file.
        if os.path.exists(output_mask_path):
            return "Skipped (already exists)"

        output_img_path = os.path.join(OUTPUT_DATA_ROOT, 'Images', 'img_dir', split, f"{base_filename}.jpg")
        
        with open(json_path, 'r') as f:
            data = json.load(f)

        height = data['size']['height']
        width = data['size']['width']
        final_mask = np.zeros((height, width), dtype=np.uint8)

        for i, obj in enumerate(data['objects']):
            original_class_id = obj['classId']
            
            # Use the mappings to get the final class ID
            class_title = original_id_to_title.get(original_class_id)
            if not class_title:
                if process_id == 0 and i < 2:
                    print(f"DEBUG: Original classId {original_class_id} not found in meta.json. Skipping.")
                continue

            new_class_id = title_to_new_id.get(class_title)
            if new_class_id is None:
                if process_id == 0 and i < 2:
                    print(f"DEBUG: Class title '{class_title}' not found in our target CLASS_NAMES. Skipping.")
                continue

            small_mask_array = base64_to_mask(obj['bitmap']['data'])
            binary_small_mask = (small_mask_array > 0)
            
            origin = obj['bitmap']['origin']
            x_start, y_start = origin[0], origin[1]
            
            small_mask_h, small_mask_w = binary_small_mask.shape
            
            x_end = x_start + small_mask_w
            y_end = y_start + small_mask_h

            # "Paint" the class ID onto the final mask
            mask_region = final_mask[y_start:y_end, x_start:x_end]
            # This ensures we only update the region that fits within the mask_region's shape
            region_h, region_w = mask_region.shape
            paint_mask = binary_small_mask[:region_h, :region_w]
            mask_region[paint_mask] = new_class_id

            if process_id == 0 and i == 0:
                print(f"DEBUG ({base_filename}): Painting object '{class_title}' (ID: {new_class_id}) of size {small_mask_w}x{small_mask_h} at ({x_start}, {y_start})")

        Image.fromarray(final_mask).save(output_mask_path)

        input_img_path = os.path.join(INPUT_DATA_ROOT, split, 'img', f"{base_filename}.jpg")
        if not os.path.exists(output_img_path):
            shutil.copy(input_img_path, output_img_path)
        
        return "Success"
    except Exception as e:
        return f"Error processing {os.path.basename(json_path)}: {e}"

def main():
    """Main function to run the preprocessing."""
    print("Starting dataset preprocessing...")
    print("IMPORTANT: If you have run this script before, please delete the")
    print(f"'{OUTPUT_DATA_ROOT}' directory for a clean run.")
    
    print("\n1/4: Setting up directories and class maps...")
    for split in ['train', 'test']:
        os.makedirs(os.path.join(OUTPUT_DATA_ROOT, 'Images', 'ann_dir', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DATA_ROOT, 'Images', 'img_dir', split), exist_ok=True)
    
    meta_json_path = os.path.join(INPUT_DATA_ROOT, 'meta.json')
    if not os.path.exists(meta_json_path):
        print(f"FATAL ERROR: meta.json not found at {meta_json_path}")
        return
        
    original_id_to_title, title_to_new_id = create_class_maps(meta_json_path)
    
    print("2/4: Gathering files to process...")
    tasks = []
    # Assign a unique ID to each task for targeted debugging prints
    for i, split in enumerate(['train', 'test']):
        ann_dir = os.path.join(INPUT_DATA_ROOT, split, 'ann')
        if not os.path.isdir(ann_dir):
            print(f"Warning: Annotation directory not found at {ann_dir}")
            continue
        for filename in sorted(os.listdir(ann_dir)):
            if filename.endswith('.jpg.json'):
                # Pass a unique ID (just the index) to each task
                tasks.append((os.path.join(ann_dir, filename), original_id_to_title, title_to_new_id, split, len(tasks)))

    if not tasks:
        print("Error: No annotation files found. Check your INPUT_DATA_ROOT path.")
        return

    print(f"Found {len(tasks)} images to process.")

    num_workers = min(12, cpu_count())
    print(f"3/4: Starting parallel processing with {num_workers} workers...")
    
    error_list = []
    success_count = 0
    skipped_count = 0

    with Pool(processes=num_workers) as pool:
        with tqdm(total=len(tasks)) as pbar:
            for result in pool.imap_unordered(process_file, tasks):
                if result == "Success":
                    success_count += 1
                elif "Skipped" in result:
                    skipped_count += 1
                else:
                    error_list.append(result)
                pbar.update(1)

    if error_list:
        print("\n--- ERRORS OCCURRED ---")
        for error in error_list[:20]:
            print(error)
        if len(error_list) > 20:
            print(f"... and {len(error_list) - 20} more errors.")
        print(f"\n{len(error_list)} files failed to process.")
    
    print("\n--- SUMMARY ---")
    print(f"Successfully processed: {success_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Failed: {len(error_list)}")
    print("4/4: Preprocessing complete!")
    print(f"New dataset is ready at: {OUTPUT_DATA_ROOT}")

if __name__ == '__main__':
    main()

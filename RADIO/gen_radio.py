import numpy as np
import random
import os
import shutil
import pickle
from tqdm import tqdm
from collections import Counter

try:
    from medmnist import OrganSMNIST
except ImportError:
    print("MedMNIST library not found. Please install it first: pip install medmnist")
    exit()

DATASET_NAME = "OrganSMNIST"
TOTAL_CLASSES_ORGAN = 11
NUM_SAMPLES_PER_TARGET_CLASS_V1 = 500
NUM_CANDIDATES_V1 = 8
CONTEXT_SIZE_V1 = 8
NUM_SAMPLES_PER_TARGET_CLASS_V2 = 500
NUM_CANDIDATES_V2 = 8
GRID_SIZE_V2 = 9
NUM_CONTEXT_IMAGES_V2 = 8

# --- 1. Data Loading and Preparation ---

def load_and_prepare_organ_data_splits(force_redownload=False):
    """
    Loads the OrganSMNIST dataset, separating it into train, validation, and test splits.

    Organizes images by class for each split and collects all original images
    and labels for potential downstream tasks like t-SNE visualization.

    Args:
        force_redownload (bool): If True, deletes the cached MedMNIST dataset
                                 to force a fresh download.

    Returns:
        tuple: A tuple containing:
            - dict: A dictionary where keys are 'train', 'val', 'test' and values
                    are lists of lists, with each inner list holding images for a
                    specific class (e.g., data_splits['train'][0] is a list of
                    all class 0 images in the training set).
            - dict: A mapping from class index (as string) to class name.
            - bool: A success flag, True if data was loaded.
            - np.ndarray: A combined NumPy array of all images from all splits.
            - np.ndarray: A combined NumPy array of all labels from all splits.
    """
    print(f"Loading {DATASET_NAME} data (train, val, test splits)...")
    if force_redownload:
        print(f"Force redownload is TRUE. Clearing cache for {DATASET_NAME}...")
        default_medmnist_root = os.path.join(os.path.expanduser("~"), ".medmnist")
        dataset_folder = os.path.join(default_medmnist_root, "organsmnist.npz")
        if os.path.exists(dataset_folder):
            os.remove(dataset_folder)
            print("Cache file removed.")

    data_splits = {}
    source_images_for_tsne = {'images': [], 'labels': []}
    class_names_map_organ = None

    for split_name in ['train', 'val', 'test']:
        print(f"\nProcessing '{split_name}' split...")
        try:
            data = OrganSMNIST(split=split_name, download=True, as_rgb=False)
            print(f"  Raw image count: {len(data.imgs)}")

            if class_names_map_organ is None and hasattr(data, 'info'):
                class_names_map_organ = data.info['label']

            images_by_class_current_split = [[] for _ in range(TOTAL_CLASSES_ORGAN)]
            if len(data.imgs) > 0:
                source_images_for_tsne['images'].append(data.imgs)
                source_images_for_tsne['labels'].append(data.labels.flatten())
                for img, label in zip(data.imgs, data.labels.flatten()):
                    images_by_class_current_split[label].append(img)
            
            data_splits[split_name] = images_by_class_current_split
            
            print(f"  Image counts per class for '{split_name}' split:")
            for i in range(TOTAL_CLASSES_ORGAN):
                class_name = class_names_map_organ.get(str(i), f"Class {i}")
                count = len(images_by_class_current_split[i])
                print(f"    - {class_name} (Idx {i}): {count} images")
                if count == 0:
                    print(f"      WARNING: No images found for this class in this split.")
        except Exception as e:
            print(f"  ERROR loading '{split_name}' split for {DATASET_NAME}: {e}")
            data_splits[split_name] = [[] for _ in range(TOTAL_CLASSES_ORGAN)]

    if class_names_map_organ is None:
        class_names_map_organ = {str(i): f"Class {i}" for i in range(TOTAL_CLASSES_ORGAN)}

    combined_orig_images = np.concatenate(source_images_for_tsne['images']) if source_images_for_tsne['images'] else np.array([])
    combined_orig_labels = np.concatenate(source_images_for_tsne['labels']) if source_images_for_tsne['labels'] else np.array([])
    
    return data_splits, class_names_map_organ, True, combined_orig_images, combined_orig_labels

# --- 2. Image Selection Helper ---

def get_random_organ_image(class_idx, images_by_class, class_names_map, used_images=None):
    """
    Retrieves a random image for a given class from a specific data split.

    Optionally ensures that the selected image has not been used before in the
    context of the current problem being generated. If no unique images are
    left, it will reuse an image.

    Args:
        class_idx (int): The index of the class to draw an image from.
        images_by_class (list): The list of lists containing images organized by class.
        class_names_map (dict): Mapping from class index to name for error messages.
        used_images (set, optional): A set of (class_idx, image_idx_in_class)
                                     tuples that have already been used.

    Returns:
        np.ndarray: The selected image.

    Raises:
        ValueError: If the specified class has no images in the provided split.
    """
    if not images_by_class[class_idx]:
        class_name = class_names_map.get(str(class_idx), f"Class {class_idx}")
        raise ValueError(f"No images available for {class_name} (Idx {class_idx}).")

    available_indices = list(range(len(images_by_class[class_idx])))
    if used_images is not None:
        used_indices_for_class = {img_idx for c, img_idx in used_images if c == class_idx}
        available_indices = [i for i in available_indices if i not in used_indices_for_class]
    
    if not available_indices:
        # If all unique images are used, fall back to picking any image
        selected_idx_in_class = random.randrange(len(images_by_class[class_idx]))
    else:
        selected_idx_in_class = random.choice(available_indices)
    
    if used_images is not None:
         used_images.add((class_idx, selected_idx_in_class))
         
    return images_by_class[class_idx][selected_idx_in_class]

def create_radio_organsmnist_dataset_v1_from_split(
    num_samples_per_target_class, 
    images_by_class_current_split, 
    class_names_map,
    split_name_for_log="unknown"
):
    """
    Generates a V1 (Progression: A, B, -> C) RADIO-like dataset from a data split.
    """
    if not any(images_by_class_current_split):
        print(f"ERROR (V1 - {split_name_for_log}): No image data provided. Cannot generate dataset.")
        return []
        
    dataset_v1 = []
    generated_problem_fingerprints = set()
    
    all_class_indices = list(range(TOTAL_CLASSES_ORGAN))
    print(f"\nGenerating V1 (Progression) dataset from '{split_name_for_log}' split...")
    
    for target_class_C_idx in tqdm(all_class_indices, desc=f"Target Classes (V1 - {split_name_for_log})"):
        if not images_by_class_current_split[target_class_C_idx]:
            continue

        samples_created = 0
        max_attempts = num_samples_per_target_class * 50 

        for _ in range(max_attempts):
            if samples_created >= num_samples_per_target_class:
                break
            
            used_images = set()
            
            possible_source_classes = [c for c in all_class_indices if c != target_class_C_idx and images_by_class_current_split[c]]
            if len(possible_source_classes) < 2:
                break

            try:
                class_A_idx, class_B_idx = random.sample(possible_source_classes, 2)
                context_images = [
                    get_random_organ_image(class_A_idx, images_by_class_current_split, class_names_map, used_images),
                    get_random_organ_image(class_B_idx, images_by_class_current_split, class_names_map, used_images),
                    get_random_organ_image(target_class_C_idx, images_by_class_current_split, class_names_map, used_images),
                    get_random_organ_image(class_A_idx, images_by_class_current_split, class_names_map, used_images),
                    get_random_organ_image(class_B_idx, images_by_class_current_split, class_names_map, used_images),
                    get_random_organ_image(target_class_C_idx, images_by_class_current_split, class_names_map, used_images),
                    get_random_organ_image(class_A_idx, images_by_class_current_split, class_names_map, used_images),
                    get_random_organ_image(class_B_idx, images_by_class_current_split, class_names_map, used_images)
                ]
                correct_target_image = get_random_organ_image(target_class_C_idx, images_by_class_current_split, class_names_map, used_images)

                distractor_class_pool = [c for c in all_class_indices if c != target_class_C_idx and images_by_class_current_split[c]]
                num_distractors = NUM_CANDIDATES_V1 - 1
                if len(distractor_class_pool) < num_distractors: continue
                chosen_distractor_classes = random.sample(distractor_class_pool, num_distractors)
                distractor_images = [get_random_organ_image(d_idx, images_by_class_current_split, class_names_map, used_images) for d_idx in chosen_distractor_classes]

                candidate_options = list(distractor_images)
                correct_idx = random.randrange(NUM_CANDIDATES_V1)
                candidate_options.insert(correct_idx, correct_target_image)
                
                problem_fingerprint = tuple(sorted(list(used_images)))
                if problem_fingerprint in generated_problem_fingerprints:
                    # This exact combination of source images has been used. Skip and try again.
                    continue
                
                generated_problem_fingerprints.add(problem_fingerprint)
                dataset_v1.append({
                    "context_images": context_images, 
                    "candidate_options": candidate_options,
                    "target_idx": correct_idx, 
                    "progression_rule_classes": (class_A_idx, class_B_idx, target_class_C_idx),
                    "target_actual_class": target_class_C_idx
                })
                samples_created += 1

            except (ValueError, IndexError):
                continue

        if samples_created < num_samples_per_target_class:
             class_name = class_names_map.get(str(target_class_C_idx), f"C{target_class_C_idx}")
             print(f"Warning (V1 - {split_name_for_log}): Only generated {samples_created}/{num_samples_per_target_class} for target {class_name}. The data split may lack diversity or uniqueness.")

    print(f"Generated {len(dataset_v1)} unique samples for Variant 1 ({split_name_for_log} split).")
    return dataset_v1


def create_radio_organsmnist_dataset_v2_from_split(
    num_samples_per_target_class, 
    images_by_class_current_split,
    class_names_map,
    split_name_for_log="unknown"
):
    """
    Generates a V2 (Same Class) RADIO-like dataset from a data split.
    """
    if not any(images_by_class_current_split):
        print(f"ERROR (V2 - {split_name_for_log}): No image data provided. Cannot generate dataset.")
        return []

    dataset_v2 = []
    generated_problem_fingerprints = set()

    all_class_indices = list(range(TOTAL_CLASSES_ORGAN))
    print(f"\nGenerating V2 (Same Class) dataset from '{split_name_for_log}' split...")
    
    for target_class_X_idx in tqdm(all_class_indices, desc=f"Target Classes (V2 - {split_name_for_log})"):
        if not images_by_class_current_split[target_class_X_idx]:
            continue
        
        samples_created = 0
        max_attempts = num_samples_per_target_class * 50
        
        for _ in range(max_attempts):
            if samples_created >= num_samples_per_target_class:
                break
                
            used_images = set()

            try:
                context_images_list = [get_random_organ_image(target_class_X_idx, images_by_class_current_split, class_names_map, used_images) for _ in range(NUM_CONTEXT_IMAGES_V2)]
                correct_candidate_image = get_random_organ_image(target_class_X_idx, images_by_class_current_split, class_names_map, used_images)
                
                context_grid = [None] * GRID_SIZE_V2
                empty_slot_idx = random.randrange(GRID_SIZE_V2)
                context_img_iter = iter(context_images_list)
                for i in range(GRID_SIZE_V2):
                    if i != empty_slot_idx:
                        context_grid[i] = next(context_img_iter)
                
                distractor_class_pool = [c for c in all_class_indices if c != target_class_X_idx and images_by_class_current_split[c]]
                num_distractors = NUM_CANDIDATES_V2 - 1
                if len(distractor_class_pool) < num_distractors: continue
                chosen_distractor_classes = random.sample(distractor_class_pool, num_distractors)
                distractor_images = [get_random_organ_image(d_idx, images_by_class_current_split, class_names_map, used_images) for d_idx in chosen_distractor_classes]

                candidate_options = list(distractor_images)
                correct_idx = random.randrange(NUM_CANDIDATES_V2)
                candidate_options.insert(correct_idx, correct_candidate_image)

                problem_fingerprint = tuple(sorted(list(used_images)))
                if problem_fingerprint in generated_problem_fingerprints:
                    continue
                
                generated_problem_fingerprints.add(problem_fingerprint)
                dataset_v2.append({
                    "context_grid": context_grid, 
                    "candidate_options": candidate_options,
                    "target_idx_in_options": correct_idx, 
                    "target_actual_class": target_class_X_idx
                })
                samples_created += 1
            
            except (ValueError, IndexError):
                continue

        if samples_created < num_samples_per_target_class:
             class_name = class_names_map.get(str(target_class_X_idx), f"C{target_class_X_idx}")
             print(f"Warning (V2 - {split_name_for_log}): Only generated {samples_created}/{num_samples_per_target_class} for target {class_name}.")

    print(f"Generated {len(dataset_v2)} unique samples for Variant 2 ({split_name_for_log} split).")
    return dataset_v2


import numpy as np
import random
import os
import pickle
from tqdm import tqdm

if __name__ == "__main__":
    # Set a fixed seed for reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    print(f"Using fixed random seed: {SEED}")

    # 1. Load OrganSMNIST data, keeping train/val/test splits separate.
    # The `data_splits_organ` dictionary will look like:
    # {'train': images_by_class_train, 'val': images_by_class_val, ...}
    data_splits_organ, class_names_map_organ, load_success, _, _ = \
        load_and_prepare_organ_data_splits(force_redownload=False)

    if not load_success:
        print(f"Halting script as {DATASET_NAME} data could not be loaded.")
        exit()

    # Dictionary to store the generated RADIO datasets for each split
    radio_datasets_v2 = {}

    # Define the proportion of the total target samples to generate for each split
    split_to_percent = {
        'train': 0.8,  # 80% of target samples for the training set
        'val': 0.1,    # 10% for validation
        'test': 0.1    # 10% for testing
    }

    # 2. Generate RADIO datasets for each source MedMNIST split
    for split_key in ['train', 'val', 'test']:
        print(f"\n{'='*25} PROCESSING SPLIT: {split_key.upper()} {'='*25}")
        
        images_by_class_current_split = data_splits_organ.get(split_key)
        
        # Validate that the current split has data to process
        if not images_by_class_current_split or all(not cls_list for cls_list in images_by_class_current_split):
            print(f"Split '{split_key}' contains no usable image data. Skipping generation.")
            radio_datasets_v2[split_key] = []
            continue

        # Calculate the number of RADIO problems to generate for this specific split
        samples_per_target_class_v2_for_split = int(
            NUM_SAMPLES_PER_TARGET_CLASS_V2 * split_to_percent[split_key]
        )
        print(f"Targeting {samples_per_target_class_v2_for_split} samples per class for this split.")

        # --- Generate Variant 2 dataset for the current split ---
        dataset_v2_split = create_radio_organsmnist_dataset_v2_from_split(
            num_samples_per_target_class=samples_per_target_class_v2_for_split,
            images_by_class_current_split=images_by_class_current_split,
            class_names_map=class_names_map_organ,
            split_name_for_log=split_key
        )
        radio_datasets_v2[split_key] = dataset_v2_split

    # 3. Save the generated datasets to separate files
    output_dir = "./generated_radio_datasets_split/"
    os.makedirs(output_dir, exist_ok=True)

    print("\n--- Saving generated datasets ---")
    for split_key, dataset in radio_datasets_v2.items():
        if dataset:
            file_path = os.path.join(output_dir, f"raven_organsmnist_v2_{split_key}.pkl")
            with open(file_path, 'wb') as f:
                pickle.dump(dataset, f)
            print(f"Saved V2 '{split_key}' dataset to {file_path} ({len(dataset)} samples)")
        else:
            print(f"No dataset was generated for the '{split_key}' split, so no file was saved.")

    print("\nDataset generation complete.")
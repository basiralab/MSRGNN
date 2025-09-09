import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import pickle
import random

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

class RadioV1Dataset(Dataset):
    def __init__(self, data_list, mode="train", transform_train=None, transform_eval=None):
        self.data_list = data_list
        self.mode = mode
        self.transform_train = transform_train
        self.transform_eval = transform_eval
        self.ctx_size = 8
        self.candidate_size = 8

        self.labels = [item['target_idx'] for item in self.data_list]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample_data = self.data_list[idx]

        context_raw = sample_data["context_images"]
        options_raw = sample_data["candidate_options"] 
        original_target_idx_in_options = sample_data["target_idx"] 

        shuffled_options_raw = list(options_raw)
        new_target_idx = original_target_idx_in_options

        if self.mode == "train":
            # Shuffle the candidate options and update the target index accordingly
            indices_for_shuffling = list(range(self.candidate_size))
            random.shuffle(indices_for_shuffling)

            # Reorder the options based on shuffled indices
            shuffled_options_raw = [options_raw[i] for i in indices_for_shuffling]
            
            new_target_idx = indices_for_shuffling.index(original_target_idx_in_options)


        all_images_np = context_raw + shuffled_options_raw

        transformed_images_list = []
        current_transform = self.transform_train if self.mode == "train" else self.transform_eval

        for img_np in all_images_np:
            if current_transform:
                transformed_images_list.append(current_transform(img_np))
            else:
                transformed_images_list.append(
                    torch.from_numpy(img_np.astype(np.float32)[np.newaxis, :, :])
                )
        
        images_tensor = torch.stack(transformed_images_list)
        target_tensor = torch.tensor(new_target_idx, dtype=torch.long)

        target_actual_class = torch.tensor(sample_data["target_actual_class"], dtype=torch.long)
        progression_rule = torch.tensor(sample_data["progression_rule_classes"], dtype=torch.long)

        return images_tensor, target_tensor, target_actual_class, progression_rule


class RadioV2Dataset(Dataset):
    def __init__(self, data_list, mode="train", transform_train=None, transform_eval=None, grid_size=9, candidate_size=8):
        self.data_list = data_list
        self.mode = mode
        self.transform_train = transform_train
        self.transform_eval = transform_eval
        self.grid_size = grid_size
        self.candidate_size = candidate_size

        self.labels = [item['target_idx_in_options'] for item in self.data_list]

    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, idx):
        sample_data = self.data_list[idx]
        context_grid_raw = sample_data["context_grid"]
        options_raw = sample_data["candidate_options"]
        target_idx = sample_data["target_idx_in_options"]

        context_images_np = [panel for panel in context_grid_raw if panel is not None]

        expected_context_size = self.grid_size - 1
        if len(context_images_np) != expected_context_size:
            raise ValueError(
                f"Error in sample {idx}: Expected {expected_context_size} context images after "
                f"filtering None, but found {len(context_images_np)}."
            )

        all_images_np = context_images_np + options_raw

        current_transform = self.transform_train if self.mode == "train" else self.transform_eval

        if current_transform is None:
            print("Warning: No transform provided. Using default ToTensor().")
            current_transform = transforms.ToTensor()
            
        transformed_images_list = [current_transform(img_np) for img_np in all_images_np]
        
        images_tensor = torch.stack(transformed_images_list)
        
        target_tensor = torch.tensor(target_idx, dtype=torch.long)

        target_actual_class = torch.tensor(sample_data["target_actual_class"], dtype=torch.long)

        return images_tensor, target_tensor, target_actual_class
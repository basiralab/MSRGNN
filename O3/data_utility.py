import os
from typing import List, Tuple
from enum import Enum

import numpy as np
import torch
from skimage import color, io, transform
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split

import albumentations as A

"""
Data utility code for O3 adapted from:
https://github.com/mikomel/sal

"""

class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @staticmethod
    def all() -> List["DatasetSplit"]:
        return [e for e in DatasetSplit]

DEFAULT_DATASET_SPLITS = DatasetSplit.all()

def to_tensor(image: np.array) -> torch.Tensor:
    image = image.astype("float32") / 255.0
    return torch.tensor(image)

def shuffle_answers(panels: np.array, target: int) -> Tuple[np.array, int]:
    indices = list(range(len(panels)))
    np.random.shuffle(indices)
    return panels[indices], indices.index(target)


class Augmentor:
    def __init__(self, transform=None):
        self.transform = transform if transform else transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3)
        ])

class DeepiqDataset(Dataset):
    ORIGINAL_PANEL_SIZE = 100
    NUM_4_PANEL_PROBLEMS = 500

    def __init__(
        self,
        dataset_root_dir: str = ".",
        splits: List[DatasetSplit] = DEFAULT_DATASET_SPLITS,
        image_size: int = 80,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        do_shuffle_panels: bool = False,
        # augmentor_factory: AugmentorFactory = AugmentorFactory.IDENTITY,
        num_panels: List[int] = (4, 5),
    ):
        test_ratio = 1.0 - train_ratio - val_ratio

        self.dataset_root_dir = dataset_root_dir
        self.image_size = image_size
        self.do_shuffle_panels = do_shuffle_panels

        # Load the full manifest of problem IDs and their correct answers
        problem_ids, correct_answers = self._load_manifest(num_panels)

        # Split the data and select the desired splits
        self.problem_ids, self.correct_answers = self._get_split_data(
            problem_ids, correct_answers, splits, train_ratio, val_ratio
        )

    def __len__(self) -> int:
        return len(self.problem_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Retrieves a single data point (image context and answer).

        Returns:
            A tuple containing:
            - A tensor of the image panels (context + choices) of shape (num_panels, 1, H, W).
            - An integer representing the index of the correct answer.
        """
        # Load the composite image file
        problem_id = self.problem_ids[idx]
        image_path = os.path.join(self.dataset_root_dir, f"{problem_id}.png")
        image = io.imread(image_path)

        # Pre-process the image
        num_panels = image.shape[1] // self.ORIGINAL_PANEL_SIZE
        image = color.rgba2rgb(image)  # Handle transparency
        image = color.rgb2gray(image)  # Convert to grayscale [0, 1] float

        # Resize the entire image strip for efficiency
        image = transform.resize(
            image,
            (self.image_size, self.image_size * num_panels),
            anti_aliasing=True,
        )

        # Split the image strip into a stack of panels
        panels = np.array(
            [
                image[:, i * self.image_size : (i + 1) * self.image_size]
                for i in range(num_panels)
            ]
        )

        # Apply augmentations and transformations
        # panels = self.augmentor.augment(panels)

        correct_answer = self.correct_answers[idx]
        if self.do_shuffle_panels:
            panels, correct_answer = shuffle_answers(panels, correct_answer)

        # Convert to tensor and add channel dimension
        panels_tensor = to_tensor(panels)
        return panels_tensor.unsqueeze(dim=1), correct_answer

    def _load_manifest(self, num_panels_to_include: List[int]) -> Tuple[List[str], List[int]]:
        """Loads all problem IDs and answers, filtering by the number of panels."""
        with open(os.path.join(self.dataset_root_dir, "answers.csv"), "r") as f:
            all_correct_answers = [int(answer) for answer in f.readlines()]

        all_problem_ids = sorted(
            [f[:4] for f in os.listdir(self.dataset_root_dir) if f.endswith(".png")]
        )

        # The DeepIQ dataset is structured with 4-panel problems first, then 5-panel
        problems_4_panel = all_problem_ids[: self.NUM_4_PANEL_PROBLEMS]
        answers_4_panel = all_correct_answers[: self.NUM_4_PANEL_PROBLEMS]
        problems_5_panel = all_problem_ids[self.NUM_4_PANEL_PROBLEMS :]
        answers_5_panel = all_correct_answers[self.NUM_4_PANEL_PROBLEMS :]

        problem_ids, correct_answers = [], []
        if 4 in num_panels_to_include:
            problem_ids.extend(problems_4_panel)
            correct_answers.extend(answers_4_panel)
        if 5 in num_panels_to_include:
            problem_ids.extend(problems_5_panel)
            correct_answers.extend(answers_5_panel)

        return problem_ids, correct_answers

    @staticmethod
    def _get_split_data(
        problem_ids: List[str],
        correct_answers: List[int],
        splits: List[DatasetSplit],
        train_ratio: float,
        val_ratio: float,
    ) -> Tuple[List[str], List[int]]:
        """Splits the data manifest into train, validation, and test sets."""
        
        # First split: separate training set from the rest (validation + test)
        ids_train, ids_val_test, ans_train, ans_val_test = train_test_split(
            problem_ids,
            correct_answers,
            train_size=train_ratio,
            stratify=correct_answers,
            random_state=42,
        )

        # Second split: separate validation and test sets from the remainder
        # The new validation ratio is relative to the remaining data
        val_ratio_of_remainder = val_ratio / (1.0 - train_ratio)
        ids_val, ids_test, ans_val, ans_test = train_test_split(
            ids_val_test,
            ans_val_test,
            train_size=val_ratio_of_remainder,
            stratify=ans_val_test,
            random_state=42,
        )

        # Collect the requested splits
        final_problem_ids, final_correct_answers = [], []
        if DatasetSplit.TRAIN in splits:
            final_problem_ids.extend(ids_train)
            final_correct_answers.extend(ans_train)
        if DatasetSplit.VAL in splits:
            final_problem_ids.extend(ids_val)
            final_correct_answers.extend(ans_val)
        if DatasetSplit.TEST in splits:
            final_problem_ids.extend(ids_test)
            final_correct_answers.extend(ans_test)

        return final_problem_ids, final_correct_answers
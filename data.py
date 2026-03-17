from collections import defaultdict

import torch
from datasets import load_dataset
from torch.utils.data import Sampler

from config import DIFFICULTY_LEVELS, EVAL_DATASET, SYSTEM_PROMPT, TRAIN_DATASET


def _format_prompt(problem: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]


def load_competition_math():
    ds = load_dataset(TRAIN_DATASET, split="train")
    ds = ds.filter(lambda ex: ex.get("level") != "Level ?")

    def transform(example):
        example["prompt"] = _format_prompt(example["problem"])
        # Normalize difficulty: the dataset stores it as 'Level 1' .. 'Level 5'
        level_str = example.get("level", "Level 1")
        example["difficulty"] = int(level_str.replace("Level ", ""))
        return example

    ds = ds.map(transform)
    return ds


def load_math500():
    ds = load_dataset(EVAL_DATASET, split="test")

    def transform(example):
        example["prompt"] = _format_prompt(example["problem"])
        level_str = example.get("level", "1")
        example["difficulty"] = int(level_str)
        return example

    ds = ds.map(transform)
    return ds


class StratifiedDifficultySampler(Sampler):
    """Yields indices grouped by difficulty level.

    All indices within a batch come from the same difficulty group.
    Groups are shuffled each epoch, and indices within each group are shuffled.
    """

    def __init__(self, dataset, batch_size: int, seed: int = 42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed

        # Build index groups by difficulty
        self.groups: dict[int, list[int]] = defaultdict(list)
        for idx in range(len(dataset)):
            level = dataset[idx]["difficulty"]
            self.groups[level].append(idx)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self._epoch if hasattr(self, "_epoch") else self.seed)

        all_batches = []
        for level in DIFFICULTY_LEVELS:
            indices = self.groups.get(level, [])
            if not indices:
                continue
            # Shuffle within this difficulty group
            perm = torch.randperm(len(indices), generator=g).tolist()
            shuffled = [indices[i] for i in perm]
            # Chunk into batches
            for start in range(0, len(shuffled), self.batch_size):
                batch = shuffled[start : start + self.batch_size]
                all_batches.append(batch)

        # Shuffle the order of batches across difficulties
        batch_perm = torch.randperm(len(all_batches), generator=g).tolist()
        for bi in batch_perm:
            yield from all_batches[bi]

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch: int):
        self._epoch = epoch


class CurriculumDifficultySampler(Sampler):
    """Yields indices in ascending difficulty order (Level 1 → 5).

    All indices within a batch come from the same difficulty group.
    Indices within each group are shuffled, but batch order follows the
    curriculum (easy-to-hard) rather than being randomized.
    """

    def __init__(self, dataset, batch_size: int, seed: int = 42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed

        # Build index groups by difficulty
        self.groups: dict[int, list[int]] = defaultdict(list)
        for idx in range(len(dataset)):
            level = dataset[idx]["difficulty"]
            self.groups[level].append(idx)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self._epoch if hasattr(self, "_epoch") else self.seed)

        for level in DIFFICULTY_LEVELS:
            indices = self.groups.get(level, [])
            if not indices:
                continue
            # Shuffle within this difficulty group
            perm = torch.randperm(len(indices), generator=g).tolist()
            shuffled = [indices[i] for i in perm]
            # Chunk into batches
            for start in range(0, len(shuffled), self.batch_size):
                batch = shuffled[start : start + self.batch_size]
                yield from batch

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch: int):
        self._epoch = epoch

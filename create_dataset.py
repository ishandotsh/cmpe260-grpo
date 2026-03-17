from datasets import Dataset, concatenate_datasets, load_dataset


def main():

    ds = load_dataset("qwedsacf/competition_math", split="train")
    ds = ds.filter(lambda ex: ex.get("level") != "Level ?")

    print(f"Total examples after filtering 'Level ?': {len(ds)}")

    subsets = []
    for level_num in range(1, 6):
        level_str = f"Level {level_num}"
        level_ds = ds.filter(lambda ex, ls=level_str: ex["level"] == ls)
        count = len(level_ds)
        print(f"  {level_str}: {count} examples", end="")

        level_ds = level_ds.shuffle(seed=42)
        level_ds = level_ds.select(range(500))
        subsets.append(level_ds)

    final = concatenate_datasets(subsets).shuffle(seed=42)
    print(f"\nFinal dataset: {len(final)} examples")

    for level_num in range(1, 6):
        level_str = f"Level {level_num}"
        n = sum(1 for ex in final if ex["level"] == level_str)
        print(f"  {level_str}: {n}")

    final.push_to_hub("ishandotsh/competition_math_reduced")
    print(f"\nPushed to https://huggingface.co/datasets/{"ishandotsh/competition_math_reduced"}")


if __name__ == "__main__":
    main()

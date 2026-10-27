from datasets import load_dataset, Dataset, DatasetDict

def prepare_datasets(dataset_name, dataset_config_name, tokenizer, sub_train=0, sub_test=0, block_size=128, token=None):
    if dataset_config_name:
        datasets = load_dataset(dataset_name, dataset_config_name, token=token)
    else:
        datasets = load_dataset(dataset_name, token=token)

    train = datasets["train"]
    test = datasets["test"]

    if sub_train > 0:
        train = Dataset.from_dict(train[:sub_train])
    if sub_test > 0:
        test = Dataset.from_dict(test[:sub_test])

    datasets = DatasetDict({"train": train, "test": test})

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    def group_texts(examples):
        concatenated_examples = {k: [item for sublist in examples[k] for item in sublist] for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        # Отрезаем хвост, не кратный block_size
        total_length = (total_length // block_size) * block_size
        
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts in chunks of max size of {block_size}",
    )

    return lm_datasets

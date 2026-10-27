from datasets import load_dataset, Dataset, DatasetDict

def prepare_datasets(dataset_name, dataset_config_name, tokenizer, sub_train=0, sub_test=0, block_size=128, token=None):
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
        desc="Running tokenizer on dataset",
    )

    def group_sentences(examples, block_size=128):
        input_ids_list = examples["input_ids"]
        attention_mask_list = examples["attention_mask"]
        token_type_ids_list = examples.get("token_type_ids", None)

        grouped_input_ids = []
        grouped_attention_mask = []
        grouped_token_type_ids = [] if token_type_ids_list is not None else None

        current_ids, current_types, current_mask = [], [], []
        current_len = 0

        for sent_ids, sent_mask, sent_types in zip(
            input_ids_list,
            attention_mask_list,
            token_type_ids_list if token_type_ids_list is not None else input_ids_list
        ):
            sent_len = len(sent_ids)
            if sent_len > block_size:
                continue

            if current_len + sent_len <= block_size:
                current_ids.extend(sent_ids)
                current_mask.extend(sent_mask)
                if token_type_ids_list is not None:
                    current_types.extend(sent_types)
                current_len += sent_len

            else:
                grouped_input_ids.append(current_ids)
                grouped_attention_mask.append(current_mask)
                if token_type_ids_list is not None:
                    grouped_token_type_ids.append(current_types)

                current_ids = sent_ids.copy()
                current_mask = sent_mask.copy()
                current_types = sent_types.copy() if token_type_ids_list is not None else []
                current_len = sent_len

        result =  {
            "input_ids": grouped_input_ids,
            "attention_mask": grouped_attention_mask
        }

        if token_type_ids_list is not None:
            result["token_type_ids"] = grouped_token_type_ids

        return result

    lm_datasets = tokenized_datasets.map(
        group_sentences,
        batched=True,
        batch_size=1000,
        num_proc=4,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    return lm_datasets

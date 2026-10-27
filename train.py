import torch
from huggingface_hub import login
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

from config import parse_args
from data import prepare_datasets

def main():
    args = parse_args()
    login(token=args.hf_token)

    print(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    print(f"Loading model: {args.model_name}")
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)

    print(f"Preparing dataset: {args.dataset_name}")
    lm_datasets = prepare_datasets(
        dataset_name=args.dataset_name,
        tokenizer=tokenizer,
        sub_train=args.sub_train,
        sub_test=args.sub_test,
        block_size=args.block_size,
        token=args.hf_token
    )

    training_args_kwargs = dict(
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        save_strategy="steps",
        eval_strategy="steps",
        logging_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.eval_steps,
        disable_tqdm=False,
        save_only_model=True,
        load_best_model_at_end=False,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        report_to="none",
        hub_model_id=args.hub_model_id,
        output_dir=args.output_dir,
        push_to_hub=True,
    )

    if args.max_steps is not None:
        training_args_kwargs["max_steps"] = args.max_steps
    elif args.num_train_epochs is not None:
        training_args_kwargs["num_train_epochs"] = args.num_train_epochs

    training_args = TrainingArguments(**training_args_kwargs)
  
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["test"],
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model and tokenizer to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Pushing model and tokenizer to HFace...")
    trainer.push_to_hub()
    print("Done!")

if __name__ == "__main__":
    main()

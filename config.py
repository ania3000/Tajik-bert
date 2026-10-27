import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="MLM training")
    
    parser.add_argument('--output_dir', type=str, default="./output", help="Directory to save checkpoints")
    parser.add_argument('--hf_token', type=str, default=None, help="HF token")

    parser.add_argument('--model_name', type=str, default="bert-base-multilingual-cased", help="Model name or path")
    parser.add_argument('--tokenizer_name', type=str, default="ania3000/tjbert-tokenizer", help="Tokenizer name or path")
    parser.add_argument('--dataset_name', type=str, default="ossetic-encoders/nctl-unlab", help="Dataset on HF")
    parser.add_argument('--hub_model_id', type=str, required=True, help="Model name for HF")

    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--eval_steps', type=int, default=200)
    parser.add_argument('--save_steps', type=int, default=200)
    parser.add_argument('--per_device_train_batch_size', type=int, default=8)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=8)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--max_steps", type=int)
    group.add_argument("--num_train_epochs", type=int)

    parser.add_argument('--sub_train', type=int, default=0, help="Subset for training (0 for full)")
    parser.add_argument('--sub_test', type=int, default=0, help="Subset for evaluation (0 for full)")
    
    parser.add_argument('--block_size', type=int, default=128, help="Maximum seq. length for grouping")

    return parser.parse_args()

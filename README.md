Using:
```bash
git clone https://github.com/ania3000/Tajik-bert.git
%cd Tajik-bert
pip install -r requirements.txt
```
To train with multilingual embeddings:
```bash
python train.py \
    --output_dir "./your-path" \
    --hf_token "your_hf_token" \
    --model_name "bert-base-multilingual-cased" \
    --tokenizer_name "ania3000/tjbert-demo" \
    --dataset_name "ossetic-encoders/nctl-unlab" \
    --hub_model_id "your-username/model-name" \
    --learning_rate 5e-5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 5 \
    --eval_steps 200 \
    --save_steps 200 \
```
To train with tajik embeddings:
```
to be continued...
```

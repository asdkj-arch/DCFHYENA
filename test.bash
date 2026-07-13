#!/bin/bash

# 创建日志目录
LOG_DIR="/public/ojsys/eye/wuwencan/lihaokai/output/test_logs/gla_test"
mkdir -p "$LOG_DIR"


datasets=(

    
)



for dataset in "${datasets[@]}"; do

    LOG_FILE="$LOG_DIR/${dataset}.log"
    


    pretrained_path="${OUTPUT_BASE}/${dataset}/checkpoints/val/accuracy.ckpt"


    python -m train wandb=null experiment=hg38/nucleotide_transformer \
        train.pretrained_model_path="$pretrained_path" \
        dataset.max_length=500 \
        model.layer.l_max=1026 \
        dataset.tokenizer_name=dual \
        model.vocab_size=12 \
        model.layer.lr=0.0006 \
        dataset.dataset_name="$dataset" \
        train.test=true 2>&1 | tee -a "$LOG_FILE"
    

done


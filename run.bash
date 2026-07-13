#!/bin/bash

# 创建日志目录
LOG_DIR="/train_logs/"
mkdir -p "$LOG_DIR"

# 定义数据集名称列表
datasets=(

    # "wgEncodeAwgTfbsBroadDnd41Ezh239875UniPk"

    
)

# 获取开始时间
START_TIME=$(date)

# 遍历所有数据集
for dataset in "${datasets[@]}"; do
    # 为每个数据集创建日志文件名
    LOG_FILE="$LOG_DIR/${dataset}.log"
    
    echo "========================================="
    echo "开始处理数据集: $dataset"
    echo "日志文件: $LOG_FILE"
    echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================="
    
    # 将开始信息写入日志文件
    echo "=========================================" >> "$LOG_FILE"
    echo "开始处理数据集: $dataset" >> "$LOG_FILE"
    echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
    echo "=========================================" >> "$LOG_FILE"

    python -m train wandb=null experiment=hg38/nucleotide_transformer \
        train.pretrained_model_path=/loss1.ckpt \
        train.pretrained_model_path_aux=/loss2.ckpt \
        dataset.max_length=500 \
        model.layer.l_max=1026 \
        dataset.tokenizer_name=dual \
        model.vocab_size=12 \
        dataset.dataset_name="$dataset" \
        train.test=false 2>&1 | tee -a "$LOG_FILE"

    PYTHON_EXIT_CODE=${PIPESTATUS[0]}
    if [ $PYTHON_EXIT_CODE -eq 0 ]; then
        echo "数据集 $dataset 处理成功"
        echo "数据集 $dataset 处理成功" >> "$LOG_FILE"
    else
        echo "警告: 数据集 $dataset 处理失败，退出码: $PYTHON_EXIT_CODE"
        echo "警告: 数据集 $dataset 处理失败，退出码: $PYTHON_EXIT_CODE" >> "$LOG_FILE"

    fi
    
    # 记录结束时间到日志文件
    echo "=========================================" >> "$LOG_FILE"
    echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
    echo "数据集 $dataset 处理完成" >> "$LOG_FILE"
    echo "=========================================" >> "$LOG_FILE"
    
    echo ""
    echo "数据集 $dataset 处理完成，结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
done


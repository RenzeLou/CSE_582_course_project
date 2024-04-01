gpu=$1
batch=$2
shot=$3
model=$4
lr=$5  # 5e-4 for small, base and large; 1e-4 for 3b

export out_dir="out/${shot}-shot/cls"
export data_dir="/scratch/rml6079/project/GEN_CLS/intent_identification/data/banking_data/${shot}-shot/cls"
export CUDA_VISIBLE_DEVICES=$gpu
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export cache_dir="/scratch/rml6079/.cache"
export TRANSFORMERS_CACHE=${cache_dir}/huggingface
export CUDA_LAUNCH_BLOCKING="0"

# note to add --overwrite_cache \ when doing the final running

export epoch=5
export lr_proj=3e-3
export out_dir="out/${shot}-shot/cls/${model}"

python run_sen_cls.py \
    --model_name_or_path ${model} \
    --do_train \
    --do_eval \
    --do_predict \
    --train_file ${data_dir}/train.csv \
    --validation_file ${data_dir}/eval.csv \
    --test_file ${data_dir}/test.csv \
    --per_device_train_batch_size ${batch} \
    --per_device_eval_batch_size ${batch} \
    --cache_dir ${cache_dir} \
    --output_dir ./${out_dir}/ \
    --overwrite_output_dir \
    --overwrite_cache \
    --learning_rate ${lr} \
    --learning_rate_proj ${lr_proj} \
    --num_train_epochs ${epoch} \
    --save_strategy no \
    --evaluation_strategy epoch \
    --seed 42 \
    --max_seq_length 1024 \
    --classifier_dropout 0.2 \
    --label_column_name category \
    --label2ids /scratch/rml6079/project/GEN_CLS/intent_identification/data/banking_data/${shot}-shot/cls/categories2id.json \
    --run_name ${shot}-shot_cls_${model}


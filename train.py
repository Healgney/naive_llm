import torch
from transformers import AutoModel, BitsAndBytesConfig

from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer
from peft import TaskType, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

from dataset_util.dataset import *
import datetime

# 定义全局变量和参数
model_name_or_path = '/root/autodl-tmp/huggingface/hub/models--THUDM--chatglm3-6b/snapshots/e9e0406d062cdb887444fe5bd546833920abd4ac'  # 模型ID或本地路径
train_data_path = 'data/zhouyi_dataset_20240118_163659.csv'  # 训练数据路径(批量生成数据集）
eval_data_path = None                                               # 验证数据路径，如果没有则设置为None
seed = 8                                                            # 随机种子
lora_rank = 16                                                      # LoRA秩
lora_alpha = 32                                                     # LoRA alpha值
lora_dropout = 0.05                                                 # LoRA Dropout率


def prepare_model():
    _compute_dtype_map = {
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16
    }

    # QLoRA 量化配置
    q_config = BitsAndBytesConfig(load_in_4bit=True,
                                  bnb_4bit_quant_type='nf4',
                                  bnb_4bit_use_double_quant=True,
                                  bnb_4bit_compute_dtype=_compute_dtype_map['bf16'])
    # 加载量化后模型
    model = AutoModel.from_pretrained(model_name_or_path,
                                      quantization_config=q_config,
                                      # device_map='cuda:0',
                                      trust_remote_code=True,
                                      revision='b098244')

    model.supports_gradient_checkpointing = True
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    kbit_model = prepare_model_for_kbit_training(model)
    target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['chatglm']

    lora_config = LoraConfig(
        target_modules=target_modules,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias='none',
        inference_mode=False,
        task_type=TaskType.CAUSAL_LM
    )
    qlora_model = get_peft_model(kbit_model, lora_config)
    qlora_model.print_trainable_parameters()

    return qlora_model

def tokenize_dataset(dataset, tokenizer):
    column_names = dataset['train'].column_names
    tokenized_dataset = dataset['train'].map(
        lambda example: tokenize_func(example, tokenizer),
        batched=False,
        remove_columns=column_names
    )
    tokenized_dataset = tokenized_dataset.shuffle(seed=seed)
    tokenized_dataset = tokenized_dataset.flatten_indices()
    return tokenized_dataset


if __name__=='__main__':

    dataset = load_dataset("csv", data_files=train_data_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                              trust_remote_code=True,
                                              revision='b098244')
    data_collator = DataCollatorForChatGLM(pad_token_id=tokenizer.pad_token_id)

    tokenized_dataset = tokenize_dataset(dataset, tokenizer)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    train_epochs = 10
    output_dir = f"/root/llm_NaiveFinetune/models/epoch{train_epochs}-{timestamp}"

    training_args = TrainingArguments(
        output_dir=output_dir,          # 输出目录
        per_device_train_batch_size=8,  # 每个设备的训练批量大小
        gradient_accumulation_steps=1,  # 梯度累积步数
        learning_rate=1e-3,             # 学习率
        num_train_epochs=train_epochs,  # 训练轮数
        lr_scheduler_type="linear",     # 学习率调度器类型
        warmup_ratio=0.1,               # 预热比例
        logging_steps=1,                # 日志记录步数
        save_strategy="steps",          # 模型保存策略
        save_steps=50,                  # 模型保存步数
        optim="adamw_torch",            # 优化器类型
        fp16=True,                      # 是否使用混合精度训练
    )

    qlora_model = prepare_model()

    trainer = Trainer(
        model=qlora_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    trainer.train()

    trainer.model.save_pretrained(output_dir)




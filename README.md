
# 🔮 ChatGLM3-6B 周易问答微调项目

本项目基于 [ChatGLM3-6B](https://github.com/THUDM/ChatGLM3) 模型，使用自制“周易”问答数据集进行指令微调，旨在构建一个具备《周易》知识问答能力的中文大语言模型。

---

## 📁 项目结构

    ├── data_gen/
    │ ├── gen_dataset.py
    │ └── zhouyi_dataset_handmade.csv
    ├── dataset_utkl/
    │ └── dataset_handmade.py
    ├── inference/
    │ └──chatbox_demo.py
    ├── train.py
    ├── README.md
    └── requirements.txt

## 🧠 模型简介：ChatGLM3-6B

ChatGLM3 是清华大学智谱 AI 发布的第三代中文大语言模型，支持中英双语、多轮对话，推理能力强，适合中文语境下的定制化微调任务。

---

## 📦 环境准备

建议配置如下：

- Python ≥ 3.9
- CUDA ≥ 11.7
- PyTorch ≥ 2.0
- Transformers ≥ 4.36
- [ChatGLM3 官方依赖](https://github.com/THUDM/ChatGLM3)

安装依赖：

```bash
pip install -r requirements.txt
```

## 📊 数据准备：周易问答数据集

数据格式采用 instruction-tuning 风格，每条样本为：

    [
      {
        "instruction": "请解释乾卦的含义。",
        "input": "",
        "output": "乾卦是《周易》六十四卦之首，象征天，代表刚健、自强不息。"
      },
      {
        "instruction": "卦象“风火家人”代表什么？",
        "input": "",
        "output": "风火家人卦象代表家庭伦理，强调内外有别、各司其职，是家庭和谐的象征。"
      }
    ]

## 🧪 模型微调（LoRA + PEFT）

本项目使用 Hugging Face 的 PEFT 工具库进行 LoRA 微调，适配大模型的低资源训练。

1. 配置文件：finetune_config.json

    {
      "model_name_or_path": "THUDM/chatglm3-6b",
      "output_dir": "output/zhouyi_lora",
      "train_file": "data/zhouyi.json",
      "per_device_train_batch_size": 2,
      "gradient_accumulation_steps": 4,
      "num_train_epochs": 3,
      "learning_rate": 5e-5,
      "fp16": true,
      "use_lora": true,
      "lora_rank": 8
    }

2. 启动微调

```bash
python train.py --config finetune_config.json
```


## 前端交互：

![周易问答示例](./assets/screenshot.jpeg)


## RAG
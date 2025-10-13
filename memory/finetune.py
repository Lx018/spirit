import argparse
import json
import os

# Argument parsing
parser = argparse.ArgumentParser(description="Fine-tune a model with customizable learning rate and max steps.")
parser.add_argument('-lr', '--learning_rate', type=float, default=2e-5, help='Learning rate for training.')
parser.add_argument('-s', '--max_steps', type=int, default=60, help='Maximum number of training steps.')
parser.add_argument('-off', '--offline', action='store_true', help='Enable offline mode.')
parser.add_argument('-p', '--model_path', type=str, default="llama38", help='Path to local model for offline mode.')
parser.add_argument('-b', '--batch_size', type=int, default=2, help='Batch size for training.')  # Added batch size argument
args = parser.parse_args()

output_dir = f"outputs/lr{args.learning_rate}_steps{args.max_steps}_b{args.batch_size}"
# --- Mode Setup ---
if args.offline:
    print("Running in OFFLINE mode.")
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    model_name = args.model_path
    local_files_only = True
else:
    print("Running in ONLINE mode.")
    model_name = "unsloth/gemma-3-12b-it-bnb-4bit"
    local_files_only = False


from datasets import Dataset
from unsloth import FastLanguageModel
import torch
from transformers import TrainingArguments
from trl import SFTTrainer

# 1. Load the data
with open('data.json', 'r') as f:
    data = json.load(f)

# 2. Create a Dataset object
dataset = Dataset.from_list(data)

# 3. Load the model
max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    local_files_only = local_files_only,
)

# 4. Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = True,
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# 5. Define the trainer
EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instruction = """你是大脑中的记忆单元调度中枢，负责处理哪些记忆会被唤醒。你需要根据聊天记录、新的消息，来筛选出哪些关于能力与偏好的记忆是对回答新的消息有帮助、或对角色应对此时此刻下一秒的情景是有用的，并将消息的序号返回到activated_memory。注意，在思考<think>中不要输出超过2000token，思考尽量精简。

    现在请根据以下规则进行筛选：
    1.为了更好扮演角色，当前情境下必须清楚的能力、偏好记忆
    2.与新的消息直接或间接有关的记忆
    3.有些记忆是在当下重要的，有些虽然重要但是回答时候并不一定需要参考因此不用选择
    4.激活的记忆总数不要超过10个
    5.若没有任何需要被激活的记忆，输出序号中输出[]空的即可

    输出格式：
    [序号，之间用逗号","隔开]
    例子:
    [2,3,19,6,8,12,37......]
    """
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for input_text, output_text in zip(inputs, outputs):
        messages = [
            {"role": "user", "content": f"{input_text}\n{instruction}"},
            {"role": "assistant", "content": output_text},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False) + EOS_TOKEN
        texts.append(text)
    return texts


trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    formatting_func = formatting_prompts_func,
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = args.batch_size,  # Use the batch size argument
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = args.max_steps,
        learning_rate = args.learning_rate,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = output_dir,
        save_strategy = "steps",
        save_steps = 100,
    ),
)

# 6. Train the model
trainer.train()

# 7. Save the model
model.save_pretrained(f"{output_dir}/final_model")
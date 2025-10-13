import argparse
import json
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from unsloth import FastLanguageModel
import torch

def main():
    parser = argparse.ArgumentParser(description="Test a fine-tuned model.")
    parser.add_argument("-o", "--model_path", type=str, default="unsloth/llama-3-8b-bnb-4bit", help="Path to the fine-tuned model.")
    parser.add_argument("-t", "--test_example_idx", type=int, default=0, help="Index of the test example in data.json.")
    args = parser.parse_args()

    # 1. Load the fine-tuned model
    max_seq_length = 2048
    dtype = None
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_path,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    try:
        model.to("cuda")
    except:
        pass

    # 2. Define the prompt format
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
你是大脑中的记忆单元调度中枢，负责处理哪些记忆会被唤醒。你需要根据聊天记录、新的消息，来筛选出哪些关于能力与偏好的记忆是对回答新的消息有帮助、或对角色应对此时此刻下一秒的情景是有用的，并将消息的序号返回到activated_memory。注意，在思考<think>中不要输出超过2000token，思考尽量精简。

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

### Input:
{}

你是大脑中的记忆单元调度中枢，负责处理哪些记忆会被唤醒。你需要根据聊天记录、新的消息，来筛选出哪些关于能力与偏好的记忆是对回答新的消息有帮助、或对角色应对此时此刻下一秒的情景是有用的，并将消息的序号返回到activated_memory。注意，在思考<think>中不要输出超过2000token，思考尽量精简。

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


### Response:
{}"""

    # 3. Load data and get test example
    with open('data.json', 'r') as f:
        data = json.load(f)
    
    if args.test_example_idx >= len(data):
        print(f"Error: test_example_idx {args.test_example_idx} is out of range for data.json which has {len(data)} examples.")
        return

    test_example = data[args.test_example_idx]
    input_text = test_example['input']
    ground_truth = test_example['output']

    # 4. Format the prompt and tokenize it
    prompt = alpaca_prompt.format(input_text, "")
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    input_ids_length = inputs["input_ids"].shape[1]

    # 5. Generate text
    outputs = model.generate(**inputs, max_new_tokens = 640, use_cache = True)
    
    # 6. Decode and print the output
    generated_tokens = outputs[:, input_ids_length:]
    prediction = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    print("--- Input ---")
    print(input_text)
    print("\n--- Prediction ---")
    print(prediction)
    print("\n--- Ground Truth ---")
    print(ground_truth)

if __name__ == "__main__":
    main()

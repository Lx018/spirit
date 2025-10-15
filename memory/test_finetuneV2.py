import argparse
import json
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from unsloth import FastLanguageModel
import torch

def main():
    parser = argparse.ArgumentParser(description="Test a fine-tuned model using chat template format.")
    parser.add_argument("-o", "--model_path", type=str, default="gemma", help="Path to the fine-tuned model.")
    parser.add_argument("-t", "--test_example_idx", type=int, default=0, help="Index of the test example in new.json.")
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
    
    FastLanguageModel.for_inference(model)
    
    try:
        model.to("cuda")
    except:
        pass

    # 2. Define the instruction (same as in finetune.py)
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

    # 3. Load data and get test example
    with open('new.json', 'r') as f:
        data = json.load(f)
        data = data["cases"]
    
    if args.test_example_idx >= len(data):
        print(f"Error: test_example_idx {args.test_example_idx} is out of range for new.json which has {len(data)} examples.")
        return

    test_example = data[args.test_example_idx]
    chat_item = test_example['chat']
    message_item = test_example['message']
    memory_item = test_example['memory']
    ground_truth_think = test_example['think']
    ground_truth_output = test_example['activated_memory']

    # 4. Format the prompt using chat template (same as in finetune.py)
    messages = [
        {"role": "user", "content": chat_item + "\n memories:" + memory_item},
        {"role": "system", "content": instruction},
        {"role": "user", "content": "next message:" + message_item},
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    # 5. Generate text
    outputs = model.generate(
        **inputs, 
        max_new_tokens=1024, 
        use_cache=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )
    
    # 6. Decode and print the output
    prediction = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Remove the input prompt from the prediction
    if prompt in prediction:
        prediction = prediction[len(prompt):].strip()

    print("=" * 80)
    print("TEST EXAMPLE #{}".format(args.test_example_idx))
    print("=" * 80)
    
    print("\n--- CHAT HISTORY ---")
    print(chat_item)
    
    print("\n--- MEMORIES ---")
    print(memory_item)
    
    print("\n--- NEW MESSAGE ---")
    print(message_item)
    
    print("\n--- PREDICTION ---")
    print(prediction)
    
    print("\n--- GROUND TRUTH THINKING ---")
    print(ground_truth_think)
    
    print("\n--- GROUND TRUTH ACTIVATED MEMORY ---")
    print(ground_truth_output)
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

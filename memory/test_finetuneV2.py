import argparse
import json
import os
import re
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from unsloth import FastLanguageModel
import torch

def extract_number_list(text):
    """Extract the last list of numbers from text like [1,2,3] or [1, 2, 3]"""
    # Find all patterns like [numbers with commas]
    pattern = r'\[(\s*\d+\s*(?:,\s*\d+\s*)*)\]'
    matches = re.findall(pattern, text)
    if matches:
        # Get the last match
        numbers_str = matches[-1]
        numbers = [int(x.strip()) for x in numbers_str.split(',') if x.strip()]
        return numbers
    return []

def calculate_precision(predicted_list, ground_truth_list):
    """Calculate precision: number of correct predictions / total predictions
    Special case: if ground truth is empty and prediction has <=2 items, return 1.0"""
    
    # Convert ground truth string to list if it's a string
    if isinstance(ground_truth_list, str):
        ground_truth_list = extract_number_list(ground_truth_list)
    
    # Special case: GT has 0 items
    if len(ground_truth_list) == 0:
        if len(predicted_list) <= 2:
            return 1.0
        else:
            return 0.0
    
    if len(predicted_list) == 0:
        return 0.0
    
    ground_truth_set = set(ground_truth_list)
    matched = sum(1 for num in predicted_list if num in ground_truth_set)
    precision = matched / len(predicted_list)
    return precision

def calculate_recall(predicted_list, ground_truth_list):
    """Calculate recall: number of ground truth items covered / total ground truth items
    Special case: if ground truth is empty and prediction has <=2 items, return 1.0"""
    
    # Convert ground truth string to list if it's a string
    if isinstance(ground_truth_list, str):
        ground_truth_list = extract_number_list(ground_truth_list)
    
    # Special case: GT has 0 items
    if len(ground_truth_list) == 0:
        if len(predicted_list) <= 2:
            return 1.0
        else:
            return 0.0
    
    predicted_set = set(predicted_list)
    covered = sum(1 for num in ground_truth_list if num in predicted_set)
    recall = covered / len(ground_truth_list)
    return recall

def calculate_f1(precision, recall):
    """Calculate F1 score: harmonic mean of precision and recall"""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def main():
    parser = argparse.ArgumentParser(description="Test a fine-tuned model using chat template format.")
    parser.add_argument("-o", "--model_path", type=str, default="gemma", help="Path to the fine-tuned model.")
    parser.add_argument("-t", "--test_example_idx", type=int, default=0, help="Starting index of the test example in new.json.")
    parser.add_argument("-n", "--num_samples", type=int, default=1, help="Number of samples to test starting from index.")
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

    # 3. Load data and get test examples
    with open('new.json', 'r') as f:
        data = json.load(f)
        data = data["cases"]
    
    end_idx = args.test_example_idx + args.num_samples
    if args.test_example_idx >= len(data):
        print(f"Error: test_example_idx {args.test_example_idx} is out of range for new.json which has {len(data)} examples.")
        return
    
    if end_idx > len(data):
        print(f"Warning: Requested {args.num_samples} samples but only {len(data) - args.test_example_idx} available from index {args.test_example_idx}.")
        end_idx = len(data)
    
    # Track metrics across all samples
    all_precisions = []
    all_recalls = []
    all_f1s = []
    
    # Test multiple samples
    for idx in range(args.test_example_idx, end_idx):
        test_example = data[idx]
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

        # 7. Extract predicted numbers and calculate metrics
        predicted_list = extract_number_list(prediction)
        ground_truth_list = extract_number_list(ground_truth_output)
        precision = calculate_precision(predicted_list, ground_truth_list)
        recall = calculate_recall(predicted_list, ground_truth_list)
        f1 = calculate_f1(precision, recall)
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)

        print("=" * 80)
        print("TEST EXAMPLE #{}".format(idx))
        print("=" * 80)
        
        print("\n--- CHAT HISTORY ---")
        print(chat_item[:500] + "..." if len(chat_item) > 500 else chat_item)
        
        print("\n--- MEMORIES ---")
        print(memory_item[:500] + "..." if len(memory_item) > 500 else memory_item)
        
        print("\n--- NEW MESSAGE ---")
        print(message_item)
        
        print("\n--- PREDICTION ---")
        print(prediction)
        
        print("\n--- PREDICTED LIST ---")
        print(predicted_list)
        
        print("\n--- GROUND TRUTH THINKING ---")
        print(ground_truth_think[:300] + "..." if len(ground_truth_think) > 300 else ground_truth_think)
        
        print("\n--- GROUND TRUTH ACTIVATED MEMORY ---")
        print(ground_truth_output)
        
        print("\n--- GROUND TRUTH LIST ---")
        print(ground_truth_list)
        
        # Calculate matched items
        matched_count = len([x for x in predicted_list if x in ground_truth_list])
        covered_count = len([x for x in ground_truth_list if x in predicted_list])
        
        print("\n--- METRICS FOR",idx,"---")
        print(f"Precision: {precision:.4f} ({matched_count}/{len(predicted_list)} correct predictions)")
        print(f"Recall:    {recall:.4f} ({covered_count}/{len(ground_truth_list)} ground truth covered)")
        print(f"F1 Score:  {f1:.4f}")
        
        print("\n" + "=" * 80 + "\n")
    
    # Print final statistics
    print("\n" + "=" * 80)
    print("FINAL STATISTICS")
    print("=" * 80)
    print(f"Total samples tested: {len(all_precisions)}")
    print(f"\nAverage Precision: {sum(all_precisions) / len(all_precisions):.4f}" if all_precisions else "N/A")
    print(f"Average Recall:    {sum(all_recalls) / len(all_recalls):.4f}" if all_recalls else "N/A")
    print(f"Average F1 Score:  {sum(all_f1s) / len(all_f1s):.4f}" if all_f1s else "N/A")
    print(f"\nMin Precision: {min(all_precisions):.4f}" if all_precisions else "N/A")
    print(f"Max Precision: {max(all_precisions):.4f}" if all_precisions else "N/A")
    print(f"\nMin Recall: {min(all_recalls):.4f}" if all_recalls else "N/A")
    print(f"Max Recall: {max(all_recalls):.4f}" if all_recalls else "N/A")
    print(f"\nMin F1 Score: {min(all_f1s):.4f}" if all_f1s else "N/A")
    print(f"Max F1 Score: {max(all_f1s):.4f}" if all_f1s else "N/A")
    print("=" * 80)

if __name__ == "__main__":
    main()

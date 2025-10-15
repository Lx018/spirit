import argparse
import json
import os
import re
from volcenginesdkarkruntime import Ark
from threading import Thread, Lock
import time
import queue

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

class APITester:
    def __init__(self, model_name, num_threads=10):
        self.client = Ark(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key="47514fc4-9e54-4846-b93a-3702a3adec2c",
        )
        self.model_name = model_name
        self.num_threads = num_threads
        self.results_lock = Lock()
        self.results = []
        self.remaining_threads = 0
        self.thread_lock = Lock()
        
    def process_single_example(self, idx, test_example, instruction):
        """Process a single test example using the API"""
        with self.thread_lock:
            self.remaining_threads += 1
        
        try:
            chat_item = test_example['chat']
            message_item = test_example['message']
            memory_item = test_example['memory']
            ground_truth_think = test_example['think']
            ground_truth_output = test_example['activated_memory']
            
            # Format the prompt using same structure as test_finetuneV2.py
            # Three messages: user (chat+memories), system (instruction), user (next message)
            messages = [
                {"role": "user", "content": chat_item + "\n memories:" + memory_item},
                {"role": "system", "content": instruction},
                {"role": "user", "content": "next message:" + message_item},
            ]
            
            # Call the API
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )
            
            prediction = completion.choices[0].message.content
            
            # Extract predicted numbers and calculate metrics
            predicted_list = extract_number_list(prediction)
            ground_truth_list = extract_number_list(ground_truth_output)
            precision = calculate_precision(predicted_list, ground_truth_list)
            recall = calculate_recall(predicted_list, ground_truth_list)
            f1 = calculate_f1(precision, recall)
            
            # Store result
            result = {
                'idx': idx,
                'chat': chat_item,
                'message': message_item,
                'memory': memory_item,
                'prediction': prediction,
                'predicted_list': predicted_list,
                'ground_truth_think': ground_truth_think,
                'ground_truth_output': ground_truth_output,
                'ground_truth_list': ground_truth_list,
                'precision': precision,
                'recall': recall,
                'f1': f1,
            }
            
            with self.results_lock:
                self.results.append(result)
            
            print(f"✓ Completed example #{idx} - P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}")
            
        except Exception as e:
            print(f"✗ Error processing example #{idx}: {e}")
            with self.results_lock:
                self.results.append({
                    'idx': idx,
                    'error': str(e),
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                })
        
        finally:
            with self.thread_lock:
                self.remaining_threads -= 1

def main():
    parser = argparse.ArgumentParser(description="Test API model using Volcengine SDK with same format as test_finetuneV2.py")
    parser.add_argument("-m", "--model_name", type=str, default="doubao-seed-1-6-flash-250828", help="Model name/endpoint ID.")
    parser.add_argument("-t", "--test_example_idx", type=int, default=0, help="Starting index of the test example in new.json.")
    parser.add_argument("-n", "--num_samples", type=int, default=1, help="Number of samples to test starting from index.")
    parser.add_argument("-w", "--num_threads", type=int, default=20, help="Number of concurrent API threads.")
    args = parser.parse_args()

    # Define the instruction (same as in finetune.py)
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

    # Load data and get test examples
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
    
    print("=" * 80)
    print(f"Testing API Model: {args.model_name}")
    print(f"Testing samples: {args.test_example_idx} to {end_idx - 1} ({end_idx - args.test_example_idx} samples)")
    print(f"Concurrent threads: {args.num_threads}")
    print("=" * 80 + "\n")
    
    # Initialize API tester
    tester = APITester(args.model_name, args.num_threads)
    
    # Create thread pool
    threads = []
    task_queue = queue.Queue()
    
    # Add all tasks to queue
    for idx in range(args.test_example_idx, end_idx):
        task_queue.put((idx, data[idx]))
    
    start_time = time.time()
    
    # Worker function
    def worker():
        while True:
            try:
                idx, test_example = task_queue.get(timeout=1)
                tester.process_single_example(idx, test_example, instruction)
                task_queue.task_done()
            except queue.Empty:
                break
    
    # Start threads
    for _ in range(args.num_threads):
        thread = Thread(target=worker, daemon=True)
        thread.start()
        threads.append(thread)
    
    # Wait for all tasks to complete
    task_queue.join()
    
    # Wait for all threads to finish
    for thread in threads:
        thread.join()
    
    elapsed_time = time.time() - start_time
    
    # Sort results by index
    tester.results.sort(key=lambda x: x['idx'])
    
    # Print individual results
    print("\n" + "=" * 80)
    print("INDIVIDUAL RESULTS")
    print("=" * 80)
    
    for result in tester.results:
        if 'error' in result:
            print(f"\n[ERROR] Example #{result['idx']}: {result['error']}")
            continue
        
        print(f"\n--- TEST EXAMPLE #{result['idx']} ---")
        print(f"Chat: {result['chat'][:200]}..." if len(result['chat']) > 200 else f"Chat: {result['chat']}")
        print(f"Message: {result['message']}")
        print(f"Predicted: {result['predicted_list']}")
        print(f"Ground Truth: {result['ground_truth_list']}")
        print(f"Metrics - P: {result['precision']:.4f}, R: {result['recall']:.4f}, F1: {result['f1']:.4f}")
    
    # Calculate statistics
    valid_results = [r for r in tester.results if 'error' not in r]
    
    if not valid_results:
        print("\n✗ No valid results to calculate statistics.")
        return
    
    all_precisions = [r['precision'] for r in valid_results]
    all_recalls = [r['recall'] for r in valid_results]
    all_f1s = [r['f1'] for r in valid_results]
    
    # Print final statistics
    print("\n" + "=" * 80)
    print("FINAL STATISTICS")
    print("=" * 80)
    print(f"Total samples tested: {len(tester.results)}")
    print(f"Valid results: {len(valid_results)}")
    print(f"Errors: {len(tester.results) - len(valid_results)}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Average time per sample: {elapsed_time / len(tester.results):.2f} seconds")
    
    print(f"\nAverage Precision: {sum(all_precisions) / len(all_precisions):.4f}")
    print(f"Average Recall:    {sum(all_recalls) / len(all_recalls):.4f}")
    print(f"Average F1 Score:  {sum(all_f1s) / len(all_f1s):.4f}")
    
    print(f"\nMin Precision: {min(all_precisions):.4f}")
    print(f"Max Precision: {max(all_precisions):.4f}")
    
    print(f"\nMin Recall: {min(all_recalls):.4f}")
    print(f"Max Recall: {max(all_recalls):.4f}")
    
    print(f"\nMin F1 Score: {min(all_f1s):.4f}")
    print(f"Max F1 Score: {max(all_f1s):.4f}")
    print("=" * 80)
    
    # Save results to file
    output_file = f"api_test_results_{args.test_example_idx}_{end_idx-1}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'args': vars(args),
            'statistics': {
                'total_samples': len(tester.results),
                'valid_results': len(valid_results),
                'errors': len(tester.results) - len(valid_results),
                'total_time': elapsed_time,
                'avg_time_per_sample': elapsed_time / len(tester.results),
                'avg_precision': sum(all_precisions) / len(all_precisions),
                'avg_recall': sum(all_recalls) / len(all_recalls),
                'avg_f1': sum(all_f1s) / len(all_f1s),
            },
            'results': tester.results
        }, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results saved to: {output_file}")

if __name__ == "__main__":
    main()

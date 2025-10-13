
import json
from sentence_transformers import SentenceTransformer, util
import argparse
import os
import numpy as np

# --- 1. ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description='Test a fine-tuned sentence transformer model.')
parser.add_argument('-o', '--model_path', type=str, required=True, help='Path to the directory where the fine-tuned model is saved.')
parser.add_argument('-i', '--data_file', type=str, default='data_mem.json', help='Path to the data file (JSON).')
parser.add_argument('-n', '--top_n', type=int, default=20, help='Number of top results to retrieve.')
parser.add_argument('-t', '--test_index', type=int, default=5, help='The 0-based index of the entry to test from the data file.')
parser.add_argument('-a', '--all', action='store_true', help='Run test on all data entries and calculate average score.')
args = parser.parse_args()

# --- CONFIGURATION ---
MODEL_PATH = args.model_path
DATA_FILE = args.data_file
TEST_INDEX = args.test_index
TOP_K = args.top_n
RUN_ALL = args.all

# --- 2. LOAD MODEL AND DATA ---
print(f"Loading fine-tuned model from {MODEL_PATH}...")
try:
    model = SentenceTransformer(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure the path is correct and the directory contains the model files.")
    exit()

print(f"Loading data from {DATA_FILE}...")
try:
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: {DATA_FILE} not found.")
    exit()

def format_result_string(test_index, chat_query, ground_truth_memories, hits, memory_pool, top_k):
    """Formats the results of a single test item into a string."""
    predicted_memories = [memory_pool[hit['corpus_id']] for hit in hits]
    ground_truth_set = set(ground_truth_memories)
    
    output = []
    output.append("="*50)
    output.append(f"RESULTS FOR ENTRY {test_index}")
    output.append("="*50)
    output.append(f"\nCHAT QUERY:\n\"{chat_query}\"")
    output.append("-" * 20)

    output.append(f"GROUND TRUTH (Correct memories):")
    if not ground_truth_memories:
        output.append("  - (No ground truth memories specified for this entry)")
    else:
        for memory in ground_truth_memories:
            emoji = '✅' if memory in predicted_memories else '❓'
            output.append(f"  {emoji} \"{memory}\"")
    output.append("-" * 20)

    output.append(f"MODEL PREDICTIONS (Top {top_k} matches):")
    if not hits:
        output.append("  - The model found no matches.")
    else:
        for hit in hits:
            corpus_id = hit['corpus_id']
            score = hit['score']
            predicted_memory = memory_pool[corpus_id]
            emoji = '✅' if predicted_memory in ground_truth_set else '❌'
            output.append(f"  {emoji} Score: {score:.4f} | Memory: \"{predicted_memory}\"")
    
    return "\n".join(output)

# --- 3. MAIN LOGIC ---
if not RUN_ALL:
    # --- 3a. SINGLE TEST MODE ---
    print(f"\n--- Preparing test for entry {TEST_INDEX} ---")
    try:
        test_item = data[TEST_INDEX]
        chat_query = test_item.get('chat')
        memory_pool = test_item.get('input')
        ground_truth_indices = test_item.get('output')
        
        if not all([chat_query, memory_pool]):
            print("Error: Test item is malformed. It must contain 'chat' and 'input' keys.")
            exit()

        ground_truth_memories = [memory_pool[i] for i in ground_truth_indices if 0 <= i < len(memory_pool)]

    except IndexError:
        print(f"Error: TEST_INDEX {TEST_INDEX} is out of bounds for the dataset of size {len(data)}.")
        exit()

    print("Encoding chat query and memory pool...")
    query_embedding = model.encode(chat_query, convert_to_tensor=True)
    pool_embeddings = model.encode(memory_pool, convert_to_tensor=True)

    print("Performing semantic search...")
    hits = util.semantic_search(query_embedding, pool_embeddings, top_k=TOP_K)[0]

    # Get the string content of the predicted memories for easier comparison
    predicted_memories = [memory_pool[hit['corpus_id']] for hit in hits]

    # Create sets for efficient lookups
    ground_truth_set = set(ground_truth_memories)
    predicted_set = set(predicted_memories)

    print("\n" + "="*50)
    print("RESULTS")
    print("="*50 + "\n")

    print(f"CHAT QUERY:\n\"{chat_query}\"")
    print("-" * 20)

    print(f"GROUND TRUTH (Correct memories):")
    if not ground_truth_memories:
        print("  - (No ground truth memories specified for this entry)")
    else:
        for memory in ground_truth_memories:
            emoji = '✅' if memory in predicted_set else '❓'
            print(f"  {emoji} \"{memory}\"")
    print("-" * 20)

    print(f"MODEL PREDICTIONS (Top {TOP_K} matches):")
    if not hits:
        print("  - The model found no matches.")
    else:
        for hit in hits:
            corpus_id = hit['corpus_id']
            score = hit['score']
            predicted_memory = memory_pool[corpus_id]
            emoji = '✅' if predicted_memory in ground_truth_set else '❌'
            print(f"  {emoji} Score: {score:.4f} | Memory: \"{predicted_memory}\"")

    print("\n" + "="*50)


else:
    # --- 3b. ALL ENTRIES MODE ---
    print(f"\n--- Running test on all {len(data)} entries ---")
    
    output_dir = 'test_results_'+MODEL_PATH
    os.makedirs(output_dir, exist_ok=True)
    print(f"Result files will be saved in '{output_dir}/'")

    all_item_scores = []

    for i, test_item in enumerate(data):
        print(f"Processing item {i+1}/{len(data)}...")
        
        chat_query = test_item.get('chat')
        memory_pool = test_item.get('input')
        ground_truth_indices = test_item.get('output', [])

        if not all([chat_query, memory_pool]):
            print(f"Warning: Skipping item {i} due to missing 'chat' or 'input'.")
            continue
        
        query_embedding = model.encode(chat_query, convert_to_tensor=True)
        pool_embeddings = model.encode(memory_pool, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, pool_embeddings, top_k=TOP_K)[0]

        ground_truth_memories = [memory_pool[j] for j in ground_truth_indices if 0 <= j < len(memory_pool)]
        result_str = format_result_string(i, chat_query, ground_truth_memories, hits, memory_pool, TOP_K)
        
        output_filename = os.path.join(output_dir, f"test_result_{i}.txt")
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(result_str)

        if not ground_truth_indices:
            continue

        predicted_corpus_ids = [hit['corpus_id'] for hit in hits]
        
        item_gt_scores = []
        for gt_index in ground_truth_indices:
            try:
                # rank is 1-based
                rank = predicted_corpus_ids.index(gt_index) + 1
                score = (TOP_K - rank + 1) / TOP_K
                item_gt_scores.append(score)
            except ValueError:
                # Ground truth not found in top K predictions
                item_gt_scores.append(0)
        
        if item_gt_scores:
            avg_item_score = np.mean(item_gt_scores)
            all_item_scores.append(avg_item_score)

    if all_item_scores:
        final_average_score = np.mean(all_item_scores)
        print("\n" + "="*50)
        print("OVERALL TEST SCORE")
        print("="*50)
        print(f"\nAverage Ranking Score: {final_average_score:.4f}")
        print(f"\nThis score is the mean of average scores for each data item.")
        print(f"A score of 1.0 means the correct memory was always the #1 result.")
        print(f"A score of 0.0 means the correct memory was never in the top {TOP_K} results.")
        print("\n" + "="*50)
    else:
        print("\nCould not calculate an overall score. No test items with ground truth were found or processed.")


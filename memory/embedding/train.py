
import json
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
import math
import logging
from datetime import datetime

# Just some basic logging
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

import argparse
from datetime import datetime

# --- 1. ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description='Fine-tune a sentence transformer model.')
parser.add_argument('-b', '--batch_size', type=int, default=16, help='Batch size for training.')
parser.add_argument('-lr', '--learning_rate', type=float, default=2e-5, help='Learning rate for the optimizer.')
parser.add_argument('-ep', '--epochs', type=int, default=1, help='Number of training epochs.')
args = parser.parse_args()

# --- CONFIGURATION ---
MODEL_NAME = './all-mpnet-base-v2'
DATA_FILE = 'data_mem.json'
# Use a formatted string for the output path that includes key hyperparameters
output_path_model_name = MODEL_NAME.split('/')[-1]
OUTPUT_PATH = f'output/retrieval-model-{output_path_model_name}-b{args.batch_size}-lr{args.learning_rate}-e{args.epochs}-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate
# The proportion of data to use for training. 1.0 means all data.
# For large datasets, you might want to start with a smaller fraction like 0.1
DATA_FRACTION = 1.0


# --- 2. LOAD AND PROCESS DATA ---
logging.info(f"Loading data from {DATA_FILE}")

try:
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    logging.error(f"Error: {DATA_FILE} not found. Please ensure the data file is in the correct directory.")
    exit()

# The sentence-transformers library expects InputExample objects.
# We will create pairs of (chat, positive_memory) for training.
train_examples = []
for item in data:
    # Adjusting for the new data format: {'chat': ..., 'input': ..., 'output': ...}
    chat = item.get('chat')
    memories = item.get('input')
    matchings = item.get('output')

    print(matchings)
    
    # Create a training example for each positive match
    for match_index in matchings:
        if 0 <= match_index < len(memories):
            positive_memory = memories[match_index]
            train_examples.append(InputExample(texts=[chat, positive_memory]))
        else:
            logging.warning(f"Skipping invalid match_index {match_index} for a memory list of size {len(memories)}")

logging.info(f"Created {len(train_examples)} training examples.")

if not train_examples:
    logging.error("No training examples were created. Please check the format of your data.json file.")
    logging.error("It should be a list of [chat, [memory1, memory2,...], [matching_index1, matching_index2,...]]")
    exit()

# For large datasets, you can uncomment the following lines to work with a subset
# import random
# random.shuffle(train_examples)
# train_examples = train_examples[:int(len(train_examples) * DATA_FRACTION)]
# logging.info(f"Using {len(train_examples)} examples for this run ({DATA_FRACTION*100}%)")


# --- 3. INITIALIZE MODEL AND DATALOADER ---
logging.info(f"Loading pre-trained model: {MODEL_NAME}")
# Load a pre-trained Sentence-Transformer model
model = SentenceTransformer(MODEL_NAME)

# The DataLoader will handle batching and shuffling
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)


# --- 4. DEFINE LOSS FUNCTION ---
# MultipleNegativesRankingLoss is a great choice for this task.
# It expects (anchor, positive) pairs and treats all other items in the batch as negatives.
train_loss = losses.MultipleNegativesRankingLoss(model=model)
logging.info("Using MultipleNegativesRankingLoss")


# --- 5. TRAIN THE MODEL ---
# The .fit method from sentence-transformers handles the training loop.
warmup_steps = math.ceil(len(train_dataloader) * NUM_EPOCHS * 0.1) # 10% of train data for warm-up
logging.info(f"Training for {NUM_EPOCHS} epochs. Warmup steps: {warmup_steps}")

model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=NUM_EPOCHS,
          optimizer_params={'lr': LEARNING_RATE},
          warmup_steps=warmup_steps,
          output_path=OUTPUT_PATH,
          show_progress_bar=True)

logging.info(f"Training complete. Model saved to {OUTPUT_PATH}")


import json
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
import math
import logging
from datetime import datetime
import argparse
import random

# Just some basic logging
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

# --- 1. ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description='Fine-tune a sentence transformer model using TripletLoss.')
parser.add_argument('-b', '--batch_size', type=int, default=16, help='Batch size for training.')
parser.add_argument('-lr', '--learning_rate', type=float, default=2e-5, help='Learning rate for the optimizer.')
parser.add_argument('-ep', '--epochs', type=int, default=1, help='Number of training epochs.')
args = parser.parse_args()

# --- CONFIGURATION ---
MODEL_NAME = './all-mpnet-base-v2'
DATA_FILE = 'data_mem.json'
output_path_model_name = MODEL_NAME.split('/')[-1]
OUTPUT_PATH = f'output/triplet-loss-model-{output_path_model_name}-b{args.batch_size}-lr{args.learning_rate}-e{args.epochs}-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate
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
# For TripletLoss, we will create triplets of (anchor, positive, negative).
# anchor = chat, positive = matched memory, negative = unmatched memory
train_examples = []
for item in data:
    chat = item.get('chat')
    memories = item.get('input')
    positive_indices = set(item.get('output'))
    
    all_indices = set(range(len(memories)))
    negative_indices = list(all_indices - positive_indices)

    # If there are no negative examples for this chat, we can't create a triplet.
    if not negative_indices:
        logging.warning(f"Skipping chat with no negative memories to sample from.")
        continue

    # Create a training example for each positive match
    for positive_index in positive_indices:
        if 0 <= positive_index < len(memories):
            positive_memory = memories[positive_index]
            
            # Randomly select one negative memory from the same context
            negative_index = random.choice(negative_indices)
            negative_memory = memories[negative_index]
            
            train_examples.append(InputExample(texts=[chat, positive_memory, negative_memory]))
        else:
            logging.warning(f"Skipping invalid positive_index {positive_index} for a memory list of size {len(memories)}")

logging.info(f"Created {len(train_examples)} training examples for TripletLoss.")

if not train_examples:
    logging.error("No training examples were created. Please check the format of your data.json file.")
    logging.error("It should be a list of {'chat': ..., 'input': [memories], 'output': [matched_indices]}")
    exit()

# For large datasets, you can uncomment the following lines to work with a subset
# random.shuffle(train_examples)
# train_examples = train_examples[:int(len(train_examples) * DATA_FRACTION)]
# logging.info(f"Using {len(train_examples)} examples for this run ({DATA_FRACTION*100}%)")


# --- 3. INITIALIZE MODEL AND DATALOADER ---
logging.info(f"Loading pre-trained model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)


# --- 4. DEFINE LOSS FUNCTION ---
# TripletLoss takes an anchor, a positive, and a negative example.
# It pushes the anchor closer to the positive and further away from the negative.
train_loss = losses.TripletLoss(model=model)
logging.info("Using TripletLoss")


# --- 5. TRAIN THE MODEL ---
warmup_steps = math.ceil(len(train_dataloader) * NUM_EPOCHS * 0.1) # 10% of train data for warm-up
logging.info(f"Training for {NUM_EPOCHS} epochs. Warmup steps: {warmup_steps}")

model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=NUM_EPOCHS,
          optimizer_params={'lr': LEARNING_RATE},
          warmup_steps=warmup_steps,
          output_path=OUTPUT_PATH,
          show_progress_bar=True)

logging.info(f"Training complete. Model saved to {OUTPUT_PATH}")

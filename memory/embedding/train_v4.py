import json
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
import torch.nn.functional as F
import logging
from datetime import datetime
import argparse
import random
from tqdm import tqdm
import math
from transformers import get_linear_schedule_with_warmup

# Just some basic logging
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

# --- 1. ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description='Fine-tune a sentence transformer model using TripletLoss with a manual optimization loop.')
parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size for training.')
parser.add_argument('-lr', '--learning_rate', type=float, default=5e-6, help='Learning rate for the optimizer.')
parser.add_argument('-ep', '--epochs', type=int, default=10, help='Number of training epochs.')
parser.add_argument('-c', '--cuda_idx', type=int, default=0, help='CUDA device index to use.')
args = parser.parse_args()

# --- CONFIGURATION ---
MODEL_NAME = './all-mpnet-base-v2'
DATA_FILE = 'chatData400/all.json'
output_path_model_name = MODEL_NAME.split('/')[-1]
OUTPUT_PATH = f'chat_output/triplet-b{args.batch_size}-lr{args.learning_rate}-e{args.epochs}'
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

train_examples = []
for item in data:
    chat = item.get('chat')
    memories = item.get('input')
    positive_indices = set(item.get('output'))
    
    all_indices = set(range(len(memories)))
    negative_indices = list(all_indices - positive_indices)

    if not negative_indices:
        logging.warning(f"Skipping chat with no negative memories to sample from.")
        continue

    for positive_index in positive_indices:
        if 0 <= positive_index < len(memories):
            positive_memory = memories[positive_index]
            negative_index = random.choice(negative_indices)
            negative_memory = memories[negative_index]
            train_examples.append(InputExample(texts=[chat, positive_memory, negative_memory]))
        else:
            logging.warning(f"Skipping invalid positive_index {positive_index} for a memory list of size {len(memories)}")

logging.info(f"Created {len(train_examples)} training examples for TripletLoss.")

if not train_examples:
    logging.error("No training examples were created. Please check the format of your data.json file.")
    exit()

# --- 3. INITIALIZE MODEL AND DATALOADER ---
logging.info(f"Loading pre-trained model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

# We need to use the model's smart_batching_collate function to create batches
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE, collate_fn=model.smart_batching_collate)

# --- 4. DEFINE LOSS FUNCTION ---
# We still use the TripletLoss class to access its configuration, like the margin.
train_loss = losses.TripletLoss(model=model)
logging.info("Using TripletLoss")

# --- 5. TRAIN THE MODEL (MANUAL LOOP) ---
device = torch.device(f"cuda:{args.cuda_idx}" if torch.cuda.is_available() else "cpu")
model.to(device)
logging.info(f"Using device: {device}")

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

num_training_steps = len(train_dataloader) * NUM_EPOCHS
warmup_steps = math.ceil(num_training_steps * 0.1)
logging.info(f"Training for {NUM_EPOCHS} epochs. Total steps: {num_training_steps}. Warmup steps: {warmup_steps}.")

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=True)
    for batch in progress_bar:
        sentence_features, _ = batch
        
        # Move batch to device
        for i in range(len(sentence_features)):
            for key in sentence_features[i]:
                sentence_features[i][key] = sentence_features[i][key].to(device)
            
        # 1. Prediction: Get sentence embeddings
        reps = [model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        rep_anchor, rep_pos, rep_neg = reps

        # 2. Calculate loss using cosine similarity, which is used for inference
        distance_pos = 1 - F.cosine_similarity(rep_anchor, rep_pos)
        distance_neg = 1 - F.cosine_similarity(rep_anchor, rep_neg)

        losses = F.relu(distance_pos - distance_neg + train_loss.triplet_margin)
        loss = losses.mean()
        
        total_loss += loss.item()
        
        # 3. Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'loss': loss.item()})
        
    avg_train_loss = total_loss / len(train_dataloader)
    logging.info(f"Epoch {epoch+1} finished. Average training loss: {avg_train_loss:.4f}")

    # Save a checkpoint after each epoch
    checkpoint_save_path = f"{OUTPUT_PATH}/epoch_{epoch+1}"
    logging.info(f"Saving checkpoint for epoch {epoch+1} to {checkpoint_save_path}")
    model.save(checkpoint_save_path)

# --- 6. SAVE THE MODEL ---
logging.info(f"Saving model to {OUTPUT_PATH}")
model.save(OUTPUT_PATH)
logging.info(f"Training complete. Model saved to {OUTPUT_PATH}")

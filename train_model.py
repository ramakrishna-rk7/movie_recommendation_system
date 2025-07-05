
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments
import torch

# Load the data
df = pd.read_csv('reviews_data.csv')
reviews = df['review'].tolist()
labels = df['label'].tolist()

# Tokenize the reviews
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
inputs = tokenizer(reviews, truncation=True, padding=True, max_length=512)

# Split into train and test sets
train_inputs, test_inputs, train_labels, test_labels = train_test_split(inputs['input_ids'], labels, test_size=0.2)

# Convert to torch tensors
train_inputs = torch.tensor(train_inputs)
test_inputs = torch.tensor(test_inputs)
train_labels = torch.tensor(train_labels)
test_labels = torch.tensor(test_labels)

# Load RoBERTa model
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results", 
    evaluation_strategy="epoch", 
    per_device_train_batch_size=16, 
    per_device_eval_batch_size=64, 
    num_train_epochs=3
)

# Create Trainer instance
trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=torch.utils.data.TensorDataset(train_inputs, train_labels),
    eval_dataset=torch.utils.data.TensorDataset(test_inputs, test_labels)
)

# Train the model
trainer.train()

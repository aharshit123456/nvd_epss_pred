import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm



# Simple Model Class
class NVDRegressionModel(nn.Module):
    def __init__(self, pretrained_model_name, config,num_numerical_features):
        super(NVDRegressionModel, self).__init__()
        # model = LlamaForCausalLM.from_pretrained("meta-llama/Bitnet-2-7b-hf")
        # self.transformer = LlamaForCausalLM.from_pretrained(pretrained_model_name, config=config)
        self.transformer = AutoModelForCausalLM.from_pretrained(pretrained_model_name, config=config)
        self.linear = nn.Linear(self.transformer.config.hidden_size + num_numerical_features, 1)

    def forward(self, input_ids, attention_mask, cvss_scores):
        # print("Input IDs shape:", input_ids.shape)
        # print("Attention mask shape:", attention_mask.shape)
        # print("Numerical features shape:", cvss_scores.shape)
        transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        cls_embedding = transformer_outputs.last_hidden_state
        # CLS token representation
        # print("CLS Embedding shape:", cls_embedding.shape)
        # print("CLS Embedding min/max:", cls_embedding.min().item(), cls_embedding.max().item())
        # print("CLS Embedding (first 5 rows):", cls_embedding[:5])
        # transformer_outputs = model.transformer(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        attention_scores = transformer_outputs.attentions  # List of attention tensors
        # for i, scores in enumerate(attention_scores):
          # print(f"Layer {i} Attention min/max: {scores.min()}, {scores.max()}")

        combined_features = torch.cat((cls_embedding, cvss_scores.unsqueeze(1)), dim=1)
        # combined_features = torch.cat((cls_embedding, cvss_scores), dim=1)  # Combine embeddings and numerical features
        output = self.linear(combined_features)
        return output.squeeze()

from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
from torch.utils.checkpoint import checkpoint

# Enable gradient checkpointing



# Training Function
def train(model, dataloader, optimizer, loss_fn, device):
    model.transformer.gradient_checkpointing_enable()
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Processing", total=len(dataloader)):
        input_ids = batch['input_ids'].to(device)
        # print("Max Token ID:", input_ids.max().item())
        # print("Min Token ID:", input_ids.min().item())

        attention_mask = batch['attention_mask'].to(device)
        cvss_scores = batch['cvss_scores'].to(device)
        targets = batch['target'].to(device)
        # print(input_ids.shape)
        # print(attention_mask.shape)
        # print(cvss_scores.shape)
        # print(targets.shape)


        print("CVSS Scores:", cvss_scores[:5])
        print("Targets:", targets[:5])

        cvss_scores = torch.nan_to_num(cvss_scores, nan=0.0, posinf=1.0, neginf=0.0)
        targets = torch.nan_to_num(targets, nan=0.0, posinf=1.0, neginf=0.0)

        optimizer.zero_grad()

    # Use autocast for mixed precision
        with autocast():
            predictions = model(input_ids, attention_mask, cvss_scores)
            loss = loss_fn(predictions, targets)
            print(loss)

        # Scale the loss and perform backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
    return total_loss / len(dataloader)

# Evaluation Function
def evaluate(model, dataloader, device):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            cvss_scores = batch['cvss_scores'].to(device)
            targets = batch['target'].to(device)

            preds = model(input_ids, attention_mask, cvss_scores)
            predictions.extend(preds.cpu().numpy())
            actuals.extend(targets.cpu().numpy())
    return predictions, actuals
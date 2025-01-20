from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer , AutoModelForSequenceClassification
# from tokenization_bitnet import BitnetTokenizer
from tqdm import tqdm
import torch
from nvd_epss_pred.notebooks.module.data.data_prep import fetch_and_process_cve_data
from nvd_epss_pred.notebooks.module.models.dataset import NVDDataset
from nvd_epss_pred.notebooks.module.models.model import NVDRegressionModel, evaluate, train
from nvd_epss_pred.notebooks.module.utils import freeze_model_layers



if __name__ == "__main__":

        # Example usage
    years = [2023, 2024]  # Specify the years you want to process
    descriptions, cvss_scores, targets = fetch_and_process_cve_data(years)

    print("Number of descriptions:", len(descriptions))
    print("Number of CVSS scores:", len(cvss_scores))
    print("Number of targets:", len(targets))




    # Hyperparameters
    # PRETRAINED_MODEL_NAME = "distilbert-base-uncased"
    PRETRAINED_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    MAX_LENGTH = 64
    BATCH_SIZE = 32
    EPOCHS = 3
    LR = 1e-5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # DEVICE = "cpu"

    # Sample data (replace with your actual dataset)
    # descriptions = ["Sample CVE description 1", "Sample CVE description 2"] * 100
    # cvss_scores = [[5.4], [7.8]] * 100  # Numerical features
    # targets = [0.5, 0.8] * 100  # Exploitability score

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(PRETRAINED_MODEL_NAME)
    config.use_cache = False  # Disable caching

    descriptions = description
    cvss_scores = score
    targets = targets

    # Tokenizer and Dataset
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    # Set an existing token as the pad token
    if tokenizer.pad_token is None:
      tokenizer.pad_token = tokenizer.eos_token  # Use the end-of-sequence token as padding

    train_desc, val_desc, train_cvss, val_cvss, train_targets, val_targets = train_test_split(
        descriptions, cvss_scores, targets, test_size=0.2, random_state=42
    )

    print("Tokenizer vocab size:", tokenizer.vocab_size)


    train_dataset = NVDDataset(train_desc, train_cvss, train_targets, tokenizer, MAX_LENGTH)
    val_dataset = NVDDataset(val_desc, val_cvss, val_targets, tokenizer, MAX_LENGTH)

    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model, Optimizer, and Loss
    model = NVDRegressionModel(PRETRAINED_MODEL_NAME, config=config, num_numerical_features=1).to(DEVICE)
    # Resize the token embeddings to match the new vocabulary size
    if '[PAD]' in tokenizer.all_special_tokens:
      model.resize_token_embeddings(len(tokenizer))

    print("Model vocab size:", model.transformer.config.vocab_size)
    # print("Max token ID in input_ids:", .max())
    # print("Model vocab size:", model.transformer.config.vocab_size)

    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    loss_fn = nn.MSELoss()

    count = 0

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    for batch in train_loader:
      count = count + 1
      input_ids = batch['input_ids']
      if input_ids.max() >= model.transformer.config.vocab_size:
        print(input_ids.max())
        raise ValueError("Token ID out of range! Check tokenizer and model vocab alignment.")

    # freeze_model_layers(model, num_layers_to_freeze=len(model.transformer.h) - 2)
    freeze_model_layers(model)

    for name, param in model.named_parameters():
      print(f"{name}: {'Trainable' if param.requires_grad else 'Frozen'}")


    # Training Loop
    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, optimizer, loss_fn, DEVICE)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}")

    # Evaluation
    predictions, actuals = evaluate(model, val_loader, DEVICE)
    print("Sample Predictions:", predictions[:5])
    print("Sample Actuals:", actuals[:5])
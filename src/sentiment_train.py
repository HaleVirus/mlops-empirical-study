import mlflow
import mlflow.pytorch
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from datasets import load_dataset
import time

def tokenize_data(tokenizer, texts, labels, max_len=64):
    encodings = tokenizer(
        texts, truncation=True, padding=True,
        max_length=max_len, return_tensors="pt"
    )
    return TensorDataset(
        encodings["input_ids"],
        encodings["attention_mask"],
        torch.tensor(labels)
    )

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct = 0, 0
    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        outputs.loss.backward()
        optimizer.step()
        total_loss += outputs.loss.item()
        preds = outputs.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            preds = outputs.logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

if __name__ == "__main__":
    # VM-safe settings
    EPOCHS = 2
    BATCH_SIZE = 8        # small batch = less RAM
    LR = 2e-5
    MAX_TRAIN = 200       # only 200 samples — runs in ~5 mins on VM
    MAX_TEST  = 50
    MAX_LEN   = 64        # shorter sequences = less memory

    device = torch.device("cpu")  # force CPU — safer on VM
    print(f"Using device: {device}")
    print("Loading dataset (subset only)...")

    dataset = load_dataset("imdb")
    train_texts  = dataset["train"]["text"][:MAX_TRAIN]
    train_labels = dataset["train"]["label"][:MAX_TRAIN]
    test_texts   = dataset["test"]["text"][:MAX_TEST]
    test_labels  = dataset["test"]["label"][:MAX_TEST]

    print("Loading DistilBERT tokenizer (lightweight)...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    print("Tokenizing...")
    train_ds = tokenize_data(tokenizer, train_texts, train_labels, MAX_LEN)
    test_ds  = tokenize_data(tokenizer, test_texts,  test_labels,  MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    mlflow.set_experiment("Sentiment-DistilBERT-IMDBv1")

    with mlflow.start_run(run_name="DistilBERT-VM-safe"):
        mlflow.log_params({
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "model": "distilbert-base-uncased",
            "max_train_samples": MAX_TRAIN,
            "max_len": MAX_LEN,
            "device": "cpu"
        })

        print("Loading DistilBERT model...")
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        ).to(device)

        optimizer = AdamW(model.parameters(), lr=LR)

        start_time = time.time()
        for epoch in range(1, EPOCHS + 1):
            print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
            tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, device)
            te_loss, te_acc = evaluate(model, test_loader, device)

            mlflow.log_metrics({
                "train_loss": tr_loss,
                "train_accuracy": tr_acc,
                "test_loss": te_loss,
                "test_accuracy": te_acc
            }, step=epoch)

            print(f"train_acc={tr_acc:.4f} | test_acc={te_acc:.4f}")

        training_time = time.time() - start_time
        mlflow.log_metric("training_time_seconds", training_time)
        mlflow.pytorch.log_model(model, "distilbert-sentiment-model")

        print(f"\nDone! Time: {training_time:.1f}s | Final test acc: {te_acc:.4f}")

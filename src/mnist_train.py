import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# ---------- Model Definition ----------
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(),
            nn.Conv2d(32, 64, 3), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

# ---------- Data ----------
def get_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test  = datasets.MNIST('./data', train=False, download=True, transform=transform)
    return (DataLoader(train, batch_size=batch_size, shuffle=True),
            DataLoader(test,  batch_size=batch_size))

# ---------- Train ----------
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

# ---------- Evaluate ----------
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            total_loss += criterion(out, y).item()
            correct += (out.argmax(1) == y).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

# ---------- Main ----------
if __name__ == "__main__":
    EPOCHS = 5
    BATCH_SIZE = 64
    LR = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    mlflow.set_experiment("MNIST-Digit-Classifier")

    with mlflow.start_run(run_name="CNN-baseline"):
        # Log hyperparameters
        mlflow.log_params({
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "optimizer": "Adam",
            "model": "CNN"
        })

        train_loader, test_loader = get_loaders(BATCH_SIZE)
        model = MNISTNet().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)

        start_time = time.time()
        for epoch in range(1, EPOCHS + 1):
            tr_loss, tr_acc = train(model, train_loader, optimizer, criterion, device)
            te_loss, te_acc = evaluate(model, test_loader, criterion, device)

            mlflow.log_metrics({
                "train_loss": tr_loss,
                "train_accuracy": tr_acc,
                "test_loss": te_loss,
                "test_accuracy": te_acc
            }, step=epoch)

            print(f"Epoch {epoch}: train_acc={tr_acc:.4f} | test_acc={te_acc:.4f}")

        training_time = time.time() - start_time
        mlflow.log_metric("training_time_seconds", training_time)

        # Save model
        mlflow.pytorch.log_model(model, "mnist-model")
        torch.save(model.state_dict(), "models/mnist_model.pt")
        print(f"\nDone! Training time: {training_time:.1f}s | Final test acc: {te_acc:.4f}")

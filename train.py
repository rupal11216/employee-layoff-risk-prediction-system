import numpy as np
import joblib
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import TensorDataset, DataLoader

from model import LayoffRNN


BASE = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

X_train = np.load(MODELS_DIR / "X_train.npy")
X_val = np.load(MODELS_DIR / "X_val.npy")
y_train = np.load(MODELS_DIR / "y_train.npy").astype(np.float32)
y_val = np.load(MODELS_DIR / "y_val.npy").astype(np.float32)

input_size = X_train.shape[2]
seq_len = X_train.shape[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LayoffRNN(input_size=input_size, hidden_size=64, num_layers=2, dropout=0.3).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

X_tr = torch.tensor(X_train, dtype=torch.float32)
y_tr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_v = torch.tensor(X_val, dtype=torch.float32)
y_v = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

train_ds = TensorDataset(X_tr, y_tr)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

EPOCHS = 60
best_auc = -1.0
best_state = None
patience = 8
no_improve = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    avg_loss = total_loss / len(train_ds)

    # validation
    model.eval()
    with torch.no_grad():
        val_logits = model(X_v.to(device)).squeeze().cpu().numpy()
        val_probs = 1 / (1 + np.exp(-val_logits))
        try:
            auc = roc_auc_score(y_val, val_probs)
        except Exception:
            auc = float("nan")

    if auc > best_auc + 1e-6:
        best_auc = auc
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}  loss={avg_loss:.6f}  val_auc={auc:.4f}")

    if no_improve >= patience:
        print("Early stopping due to no improvement.")
        break

if best_state is not None:
    model.load_state_dict(best_state)

torch.save(model.state_dict(), MODELS_DIR / "best_model.pt")
print(f"Saved best model (val_auc={best_auc:.4f}).")

# calibrator on validation logits
model.eval()
with torch.no_grad():
    val_logits = model(X_v.to(device)).squeeze().cpu().numpy()
calibrator = LogisticRegression(max_iter=500)
calibrator.fit(val_logits.reshape(-1, 1), y_val)
joblib.dump(calibrator, MODELS_DIR / "calibrator.pkl")
print("Saved calibrator (calibrator.pkl).")

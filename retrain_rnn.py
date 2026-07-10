# src/retrain_rnn.py
import sys
from pathlib import Path
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.model import LayoffRNN

BASE = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE / "models"

X_train = np.load(MODELS_DIR / "X_train.npy")
y_train = np.load(MODELS_DIR / "y_train.npy").astype(np.float32)
X_val = np.load(MODELS_DIR / "X_val.npy")
y_val = np.load(MODELS_DIR / "y_val.npy").astype(np.float32)

input_size = X_train.shape[2]

model = LayoffRNN(input_size=input_size, hidden_size=64, num_layers=2, dropout=0.3)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

X_t = torch.tensor(X_train, dtype=torch.float32)
y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_v = torch.tensor(X_val, dtype=torch.float32)
y_v = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

EPOCHS = 30
for e in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    out = model(X_t)
    loss = criterion(out, y_t)
    loss.backward()
    optimizer.step()
    if (e + 1) % 5 == 0:
        print(f"Epoch {e+1}: loss={loss.item():.6f}")

# learn calibration on validation logits (safer than arbitrary shift)
model.eval()
with torch.no_grad():
    val_logits = model(X_v).squeeze().numpy()

calibrator = LogisticRegression(max_iter=300)
calibrator.fit(val_logits.reshape(-1, 1), y_val)

# derive scale & bias from calibrator (we'll apply to final fc weights/bias)
scale = float(calibrator.coef_[0][0])
bias = float(calibrator.intercept_[0])

applied = False
for name, param in model.named_parameters():
    if name.endswith("bias") and param.numel() == 1:
        with torch.no_grad():
            # scale the final bias and add learned bias (safe in-place)
            param.mul_(scale)
            param.add_(bias)
        applied = True
        print(f"✅ Applied scale={scale:.4f} and bias={bias:.4f} to parameter '{name}'")
        break

if not applied:
    print("⚠️ Could not find scalar bias to adjust.")

torch.save(model.state_dict(), MODELS_DIR / "best_model.pt")
joblib.dump(calibrator, MODELS_DIR / "calibrator.pkl")
print("💾 Saved recalibrated model and calibrator.")

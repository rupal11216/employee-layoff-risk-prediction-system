import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import json

def build_sequences(csv_path, out_dir, seq_len=6, step=1, test_frac=0.15, val_frac=0.10, random_seed=42):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = df.sort_values(['EmployeeID', 'RecordMonth'])
    
    # numeric & categorical columns (explicit)
    num_cols = ['TenureYears','Performance','TrainingHours','Projects','OvertimeHours','LastPromotionYears','Salary']
    cat_cols = ['Department']

    # encode department with OneHotEncoder (save encoder)
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    dept_ohe = enc.fit_transform(df[['Department']].fillna("").astype(str))
    dept_cols = [f"DEPT_{c}" for c in enc.categories_[0]]
    dept_df = pd.DataFrame(dept_ohe, columns=dept_cols, index=df.index)

    df = pd.concat([df.reset_index(drop=True), dept_df.reset_index(drop=True)], axis=1)

    features = num_cols + dept_cols

    X, y = [], []
    for emp, g in df.groupby('EmployeeID'):
        arr = g[features].values.astype(float)
        # take per-employee label (assumes stable per-employee)
        label = int(g['Layoff'].iloc[0])
        if arr.shape[0] < seq_len:
            continue
        for i in range(0, arr.shape[0] - seq_len + 1, step):
            X.append(arr[i:i+seq_len])
            y.append(label)

    X = np.array(X)
    y = np.array(y).astype(np.int64)

    # shuffle
    rng = np.random.RandomState(random_seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    n_test = int(len(X) * test_frac)
    n_val = int(len(X) * val_frac)
    X_test, y_test = X[:n_test], y[:n_test]
    X_val, y_val = X[n_test:n_test + n_val], y[n_test:n_test + n_val]
    X_train, y_train = X[n_test + n_val:], y[n_test + n_val:]

    # scale numeric features (first ns columns in features)
    ns = len(num_cols)
    scaler = StandardScaler()
    train_nums = X_train[:, :, :ns].reshape(-1, ns)
    scaler.fit(train_nums)

    def transform_set(Xset):
        Xt = Xset.copy()
        b, t, f = Xt.shape
        Xt2 = Xt.reshape(-1, f)
        Xt2[:, :ns] = scaler.transform(Xt2[:, :ns])
        return Xt2.reshape(b, t, f)

    X_train = transform_set(X_train)
    X_val = transform_set(X_val)
    X_test = transform_set(X_test)

    # save
    np.save(out_dir / 'X_train.npy', X_train)
    np.save(out_dir / 'y_train.npy', y_train)
    np.save(out_dir / 'X_val.npy', X_val)
    np.save(out_dir / 'y_val.npy', y_val)
    np.save(out_dir / 'X_test.npy', X_test)
    np.save(out_dir / 'y_test.npy', y_test)

    joblib.dump(scaler, out_dir / 'scaler.pkl')
    joblib.dump(enc, out_dir / 'dept_encoder.pkl')

    # save meta
    meta = {
        "seq_len": seq_len,
        "num_cols": num_cols,
        "dept_cols": dept_cols,
        "features": features,
        "input_size": len(features)
    }
    joblib.dump(meta, out_dir / 'train_meta.pkl')

    print("Saved preprocessed arrays and artifacts to", out_dir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to employee CSV")
    p.add_argument("--out", default="./models", help="Output directory for preprocessed artifacts")
    p.add_argument("--seq_len", type=int, default=6)
    args = p.parse_args()
    build_sequences(args.csv, args.out, seq_len=args.seq_len)

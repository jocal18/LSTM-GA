# ============================ #
# LSTM + GA for Time Series    #
# Clean, robust, no-leak setup #
# ============================ #

import os, math, random, time
import numpy as np
import pandas as pd

# Matplotlib: try interactive, fallback to non-interactive (no errors either way)
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler, MinMaxScaler  # switch to MinMaxScaler if you prefer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

from deap import base, creator, tools, algorithms

# ----------------
# Repro + Device
# ----------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("medium")

# ----------------
# Data Loading
# ----------------
csv_path = "./data/airline-passengers.csv"
df = pd.read_csv(csv_path)

# Support either 'Date' or 'Month' for timestamp
if "Date" in df.columns:
    date_col = "Date"
elif "Month" in df.columns:
    date_col = "Month"
else:
    raise ValueError("CSV must contain a 'Date' or 'Month' column.")

df[date_col] = pd.to_datetime(df[date_col])
df = df.set_index(df[date_col]).drop(columns=[date_col]).sort_index()

# target: first numeric column
num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
if len(num_cols) == 0:
    raise ValueError("No numeric target column found in CSV.")
target_col = num_cols[0]
series = df[target_col].astype("float32").values  # (T,)

# ----------------
# Windowing
# ----------------
look_back = 20  # window length

def make_windows(arr: np.ndarray, L: int):
    X, y = [], []
    for i in range(len(arr) - L):
        X.append(arr[i:i+L])
        y.append(arr[i+L])
    X = np.asarray(X, dtype=np.float32)[..., None]  # (N, L, 1)
    y = np.asarray(y, dtype=np.float32)             # (N,)
    return X, y

X_all_raw, y_all_raw = make_windows(series, look_back)
N = len(y_all_raw)

# TimeSeriesSplit for an out-of-sample test
tscv = TimeSeriesSplit(n_splits=2)
for tr_idx, te_idx in tscv.split(np.arange(N)):
    pass  # use the last split
X_train_raw, y_train_raw = X_all_raw[tr_idx], y_all_raw[tr_idx]
X_test_raw,  y_test_raw  = X_all_raw[te_idx],  y_all_raw[te_idx]

# ----------------
# Scaling (fit on TRAIN only) – StandardScaler; swap to MinMaxScaler if desired
# ----------------
scaler = MinMaxScaler(feature_range=(0, 1))
train_vals = np.concatenate([X_train_raw.reshape(-1), y_train_raw]).reshape(-1, 1)
scaler.fit(train_vals)

def scale1d(a: np.ndarray) -> np.ndarray:
    return scaler.transform(a.reshape(-1, 1)).reshape(a.shape)

def inv_scale1d(a: np.ndarray) -> np.ndarray:
    return scaler.inverse_transform(a.reshape(-1, 1)).reshape(a.shape)

X_train = scale1d(X_train_raw)
y_train = scale1d(y_train_raw)
X_test  = scale1d(X_test_raw)
y_test  = scale1d(y_test_raw)

# Torch tensors (inputs/targets DO NOT require grad)
X_train_t = torch.from_numpy(X_train)
y_train_t = torch.from_numpy(y_train)
X_test_t  = torch.from_numpy(X_test)
y_test_t  = torch.from_numpy(y_test)

# Small val split from the training tail (for GA fitness)
val_frac = 0.15
cut = int(len(X_train_t) * (1 - val_frac))
X_tr, X_val = X_train_t[:cut], X_train_t[cut:]
y_tr, y_val = y_train_t[:cut], y_train_t[cut:]

# ----------------
# Model
# ----------------
class LSTMModel(nn.Module):
    def __init__(self, hidden_size=64, num_layers=1, dropout=0.1, input_size=1, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, output_size)
        self._init_weights()

    def _init_weights(self):
        # safe init with forget-gate bias = 1.0 on input-side biases
        with torch.no_grad():
            for name, p in self.lstm.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(p)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(p)
                elif "bias_ih" in name:
                    p.fill_(0.0)
                    h = p.shape[0] // 4  # [i, f, g, o]
                    p[h:2*h].fill_(1.0)  # forget gate bias
                elif "bias_hh" in name:
                    p.zero_()
            nn.init.xavier_uniform_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, x):
        # x: (B, T, 1)
        out, _ = self.lstm(x)          # (B, T, H)
        h_last = out[:, -1, :]         # (B, H)
        h_last = self.dropout(h_last)
        y = self.head(h_last).squeeze(-1)  # (B,)
        return y

# ----------------
# Training utilities
# ----------------
def make_optimizer(name: str, params, lr: float):
    name = name.lower()
    if name == "adam":    return torch.optim.Adam(params, lr=lr)
    if name == "rmsprop": return torch.optim.RMSprop(params, lr=lr)
    if name == "sgd":     return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
    if name == "adamax":  return torch.optim.Adamax(params, lr=lr)
    if name == "adagrad": return torch.optim.Adagrad(params, lr=lr)
    raise ValueError(f"Unknown optimizer: {name}")

def train_model(model, X, y, epochs, optimizer_name, lr, batch_size=64, clip=1.0):
    model.to(device)
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=(device.type=="cuda"))
    opt = make_optimizer(optimizer_name, model.parameters(), lr)
    loss_fn = nn.MSELoss()
    scaler_amp = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                pred = model(xb)
                loss = loss_fn(pred, yb)
            scaler_amp.scale(loss).backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler_amp.step(opt)
            scaler_amp.update()

@torch.no_grad()
def val_loss(model, X, y):
    model.eval()
    X = X.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    pred = model(X)
    return nn.functional.mse_loss(pred, y).item()

@torch.no_grad()
def predict(model, X, bs=512):
    model.eval()
    preds = []
    for i in range(0, len(X), bs):
        xb = X[i:i+bs].to(device, non_blocking=True)
        preds.append(model(xb).cpu())
    return torch.cat(preds, dim=0).numpy()

# ----------------
# Genetic Algorithm (optimize val loss)
# ----------------
# In notebooks, creator.* must be defined once; script is safe as-is:
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Search space: sensible bounds to avoid "flat" solutions
toolbox.register("units",       random.randint, 32, 128)
toolbox.register("num_layers",  random.randint, 1, 2)
toolbox.register("dropout",     random.uniform, 0.0, 0.2)
toolbox.register("optimizer",   random.choice,  ["Adam", "RMSprop", "SGD"])
toolbox.register("learning_rate", lambda: 10 ** random.uniform(-4, -2))  # log-uniform [1e-4, 1e-2]

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.units, toolbox.num_layers, toolbox.dropout,
                  toolbox.optimizer, toolbox.learning_rate), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def eval_ind(ind):
    units, layers, drop, opt, lr = ind
    model = LSTMModel(hidden_size=int(units),
                      num_layers=int(layers),
                      dropout=float(drop))
    # modest epochs for GA fitness; validation decides fitness
    epochs = 50
    train_model(model, X_tr, y_tr, epochs=epochs, optimizer_name=opt, lr=float(lr))
    vloss = val_loss(model, X_val, y_val)
    # fitness: larger is better
    return (1.0 / (1.0 + vloss),)

def mutate_ind(ind, p=0.25):
    if random.random() < p: ind[0] = toolbox.units()
    if random.random() < p: ind[1] = toolbox.num_layers()
    if random.random() < p: ind[2] = toolbox.dropout()
    if random.random() < p: ind[3] = toolbox.optimizer()
    if random.random() < p: ind[4] = toolbox.learning_rate()
    return (ind,)

toolbox.register("evaluate", eval_ind)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutate_ind)
toolbox.register("select", tools.selTournament, tournsize=3)

pop_size, ngen, cxpb, mutpb = 12, 12, 0.6, 0.3
pop = toolbox.population(n=pop_size)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values[0])
stats.register("avg", np.mean); stats.register("max", np.max)

pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb,
                                   ngen=ngen, stats=stats, halloffame=hof, verbose=True)

best = hof[0]
print("Best individual (units, layers, dropout, opt, lr):", best)

# ----------------
# Retrain best on full TRAIN (tr+val) for more epochs
# ----------------
units, layers, drop, opt, lr = best
best_model = LSTMModel(hidden_size=int(units),
                       num_layers=int(layers),
                       dropout=float(drop))
train_model(best_model, X_train_t, y_train_t, epochs=150, optimizer_name=opt, lr=float(lr))

# ----------------
# Evaluation on TEST (inverse-scaled)
# ----------------
y_pred_test_scaled = predict(best_model, X_test_t)
y_pred_test = inv_scale1d(y_pred_test_scaled)
y_test_orig = y_test_raw  # already unscaled by construction

mse = mean_squared_error(y_test_orig, y_pred_test)
mae = mean_absolute_error(y_test_orig, y_pred_test)
eps = np.finfo(np.float32).eps
mape = np.mean(np.abs((y_test_orig - y_pred_test) / np.maximum(np.abs(y_test_orig), eps))) * 100.0
print(f"MSE:  {mse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"MAPE: {mape:.2f}%")

# Directional accuracy
def direction(arr):
    return (np.diff(arr) >= 0).astype(np.int8)
da = (direction(y_pred_test) == direction(y_test_orig)).mean()
print(f"Directional accuracy: {da:.3f}")

# Naive baseline: last value in window
y_naive = X_test_raw[:, -1, 0]
print("Naive MAE:", mean_absolute_error(y_test_orig, y_naive))

# ----------------
# Plotting (save always; show only if interactive)
# ----------------
def plot_predictions(dates, y_true, y_pred, title, fname="lstm_ga_forecast.png"):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, y_true, label="Actual")
    ax.plot(dates, y_pred, label="Predicted")
    ax.set_xlabel("Date"); ax.set_ylabel(target_col); ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(fname, dpi=150)
    # show only if backend is interactive
    backend = matplotlib.get_backend().lower()
    if any(k in backend for k in ["tkagg", "qt", "macosx"]):
        plt.show()
    plt.close(fig)
    print(f"Saved plot to: {fname}")

# Align test dates: targets start at index look_back
test_dates = df.index[look_back + te_idx]
plot_predictions(test_dates, y_test_orig, y_pred_test, "LSTM + GA (train-only scaling)")

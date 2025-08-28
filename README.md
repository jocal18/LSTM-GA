# LSTM + Genetic Algorithm for Stock Price Prediction

A PyTorch-based LSTM trained on time-series windows of stock prices, with hyperparameters optimized by a Genetic Algorithm (DEAP). The project includes data preprocessing with `scikit-learn`, time-series splits, GPU support, and evaluation/visualization utilities.

---

## ✨ What This Repo Does

- Loads historical OHLCV data (e.g., `MSFT.csv`) and standardizes features  
- Builds rolling input windows (`look_back`) for supervised training  
- Defines an LSTM model with:
  - Linear → ReLU → LSTM stack
  - Configurable `hidden_size`, `num_layers`, `dropout`
  - Careful LSTM weight initialization (Kaiming/Orthogonal/Zero-bias)  
- Uses **DEAP** to **maximize** fitness (1 / (1 + MSE)) across hyperparameters:
  - `hidden_size` (units)
  - `dropout_rate`
  - `optimizer` (Adagrad/Adam/Adamax/RMSprop/SGD)
  - `learning_rate`
  - `num_layers`
- Trains with early-stopping heuristic when the loss plateaus
- Evaluates on a final time-series split and plots **actual vs. predicted** prices
- Reports direction-of-movement hit rate, MSE, MAE (and MAPE, see note below)

---

## 🧱 Project Structure

```
.
├── README.md
├── requirements.txt
├── data/
│   └── MSFT.csv                 # Your input data (see format below)
└── src/
    └── main.py                  # (Your provided script; can be named as you wish)
```

> If you keep everything in a single file, just place it at repo root (e.g., `main.py`).

---

## 📦 Requirements

Create a fresh environment (Python 3.10+ recommended) and install:

```txt
matplotlib
pandas
numpy
torch         # + torchvision/torchaudio if desired
scikit-learn
deap
```

Example:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # or pip install matplotlib pandas numpy torch scikit-learn deap
```

> GPU is optional. If CUDA is available, PyTorch will run on `cuda:0` automatically.

---

## 📁 Input Data Format

Place your CSV at `./data/MSFT.csv` with a **Date** column and at least one numeric feature. Example minimal schema:

| Date       | Close | Open | High | Low | Volume |
|------------|-------|------|------|-----|--------|
| 2020-01-02 | 160.6 | ...  | ...  | ... | ...    |

- The script:
  - Parses `Date` and sets it as the index
  - Standard-scales **all numeric columns**
  - Uses column index `INDEX_X` for inputs and `INDEX_Y` for the target (default `0`, i.e., first numeric column after Date)
- Adjust `INDEX_X` / `INDEX_Y` to target another column (e.g., `Close`).

---

## 🚀 Quickstart

1. Put `MSFT.csv` in `./data/`
2. (Optional) Edit hyperparameters and GA settings in the script
3. Run:

```bash
python src/main.py
# or: python main.py
```

You’ll see:
- GA progress logs across generations  
- Final selected hyperparameters  
- A plot comparing actual vs. predicted prices on the test segment  
- Metrics printed to stdout  

---

## ⚙️ Key Configuration Knobs

Inside the script:

- **Windowing**
  - `look_back = 14` (timesteps per input sample)
- **Target/Input Columns**
  - `INDEX_Y = 0` (target)
  - `INDEX_X = 0` (input feature index)
- **Scaler**
  - `StandardScaler()` (can swap to `MinMaxScaler` if preferred)
- **Time-Series Split**
  - `TimeSeriesSplit(n_splits=2)` (train/test folding)
- **Genetic Algorithm**
  - Population size: `population = toolbox.population(n=4)`
  - Generations: `ngen = 20`
  - Crossover prob: `cxpb=0.5`
  - Mutation prob: `mutpb=0.3`
  - Mutation probabilities per gene in `custom_mutation(...)`
- **Per-Evaluation Training**
  - `epoch=3` during GA (kept low for speed)
- **Final Training**
  - `epochs=300` on best individual

> Tip: Increase `population`, `ngen`, and GA train `epoch` once the pipeline works to search more thoroughly.

---

## 🧠 Model Architecture

```
Input (B, T, 1)
   └─ Linear(in=1, out=hidden_size)
      └─ ReLU
         └─ LSTM(hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
            └─ Take h_n (all layers), reshape to (B, num_layers * hidden_size)
               └─ Dropout(dropout_rate)
                  └─ Linear(in=num_layers*hidden_size, out=1) → prediction
```

- Weight init:
  - `weight_ih`: Kaiming normal
  - `weight_hh`: Orthogonal
  - `bias`: zeros

---

## 🔬 Fitness & Evaluation

- **GA Fitness**: `1 / (1 + epoch_loss)` where `epoch_loss` is average MSE over training batches  
- **Metrics on Test Set**:
  - `MSE`, `MAE`
  - **Direction Accuracy**: compares `np.diff` sign between actual and predicted
  - **MAPE**: see **Notes** (there’s a bug in the raw formula as written)

- **Unscaling** predictions: the helper `unscal_data` reconstructs a multi-feature array with zeros except the target column, then calls `scaler.inverse_transform` and extracts the target back. This ensures inverse scaling is consistent with multi-feature scaling.

---

## 📊 Output

- A Matplotlib plot: **Actual Prices** vs. **Predicted Prices** on the test segment
- Console logs with:
  - GA evolution
  - Final metrics (MSE/MAE/MAPE) and movement hit rate
  - Best individual (hyperparameters)

> The script uses `TkAgg` as Matplotlib backend. If you run in a headless environment (e.g., remote server), consider switching to `Agg` and saving figures instead:
> ```python
> import matplotlib
> matplotlib.use('Agg')
> plt.savefig('prediction.png', dpi=200, bbox_inches='tight')
> ```

---

## 🧪 Reproducibility

For reproducible runs, set seeds for `random`, `numpy`, and `torch` at the top of the script:

```python
import os, random, numpy as np, torch
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## ✅ Usage Tips & Best Practices

- Start small (fewer generations & population) to validate your pipeline.
- Once stable, scale up GA search and/or evaluation epochs.
- Monitor overfitting:
  - Use a larger `TimeSeriesSplit` (`n_splits>=3`) if your dataset is long enough.
  - Consider adding validation early stopping (separate from GA).
- Try more informative features:
  - Log returns, rolling means/volatility, technical indicators (RSI/MACD), etc.
  - Update `INDEX_X`/`INDEX_Y` or stack multiple inputs (adjust input size & Linear layer).

---

## 🛠️ Known Quirks & Improvements

- **MAPE Formula Bug**  
  The current line:
  ```python
  mape_test = np.mean(np.abs((y_test, y_predicted) / y_test)) * 100
  ```
  should be:
  ```python
  mape_test = np.mean(np.abs((y_test - y_predicted) / y_test)) * 100
  ```
  (Also guard against division by zero.)

- **Selection Pressure**  
  The GA uses `tools.selBest` for selection, which can reduce diversity. Consider `selTournament`, `selRoulette`, or `mu+lambda` strategies for better exploration.

- **Batch Shuffling**  
  DataLoader uses `shuffle=False` to respect temporal ordering. Keep it that way for time-series; if you add random windows or rolling batches, document the rationale.

- **Early Stopping Heuristic**  
  The current plateau check uses an absolute delta (`< 0.01`) for **50** epochs. You may prefer patience-based validation loss early stopping.

- **Multi-Feature Inputs**  
  Code assumes a single feature per timestep (`input_size=1`). To use multiple features, expand the input tensor’s last dimension and update `linear_1 = nn.Linear(input_size, hidden_size)` accordingly.

---

## 🧩 Troubleshooting

- **“No module named …”**:  
  Make sure you’re in the virtual environment and installed all requirements.
- **Matplotlib/Display Errors**:  
  On headless servers, switch backend to `Agg` and `plt.savefig(...)` instead of `plt.show()`.
- **CUDA Out of Memory**:  
  Reduce batch size, hidden units, or number of layers. Ensure other GPU processes are stopped.
- **Inverse Scaling Shape Errors**:  
  Ensure the scaler was fit on the same number of features you reconstruct in `unscal_data`.

---

## 📈 Extending This Repo

- Add CLI args with `argparse` (for paths, look_back, GA settings, etc.).
- Log results to MLFlow/W&B; save artifacts (best hyperparams, plots, predictions).
- Add a proper validation fold separate from test for GA fitness (avoid mild leakage).
- Implement rolling/expanding backtests over multiple train/test windows.
- Package as a module with tests (`pytest`) and CI.

---

## 📝 License

MIT (or your preferred license). Add `LICENSE` file to the repo root.

---

## 🙌 Acknowledgements

- [PyTorch](https://pytorch.org/)
- [DEAP](https://deap.readthedocs.io/)
- [scikit-learn](https://scikit-learn.org/)
- [Matplotlib](https://matplotlib.org/)

---

## 🔧 Minimal `requirements.txt`

```txt
matplotlib>=3.7
pandas>=2.0
numpy>=1.24
torch>=2.2
scikit-learn>=1.3
deap>=1.3
```

---

## 💡 Example: Adjusting Hyperparameter Ranges

Inside the GA setup:

```python
toolbox.register('units',         random.uniform, 16, 256)         # wider search
toolbox.register('dropout_rate',  random.uniform, 0.0, 0.6)
toolbox.register('optimizer',     random.choice, ['Adam', 'RMSprop', 'SGD'])
toolbox.register('learning_rate', random.uniform, 1e-5, 1e-2)
toolbox.register('num_layers',    random.uniform, 1, 3)
```

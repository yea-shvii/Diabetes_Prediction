# Diabetes Risk Prediction Pipeline

A end-to-end machine learning pipeline that trains a neural network on synthetic patient data and tests it on real-world data from the Pima Indians Diabetes Dataset (Kaggle). Built as a hands-on learning project covering NumPy, Pandas, Matplotlib, and PyTorch.

---

## Project Structure

```
diabetes_pred/
│
├── diabetes_pipeline.ipynb       # Main notebook — data generation, training, saving
├── test_on_real_data.ipynb       # Inference notebook — loads model, tests on Kaggle data
├── diabetes.csv                  # Kaggle dataset (download separately)
└── README.md
```

---

## What This Project Does

| Stage | Description |
|---|---|
| Data Generation | Creates 1,000 synthetic patient records using NumPy |
| Data Cleaning | Handles missing values, outliers, type casting with Pandas |
| Feature Engineering | Adds BMI categories, age groups via `apply` and `groupby` |
| Visualization | 4 charts — histogram, scatter, bar, line using Matplotlib |
| Model Training | Trains a 3-layer neural network using PyTorch |
| Real-world Testing | Loads saved model and runs inference on 768 real patients |

---

## Tech Stack

- **NumPy** — array creation, boolean indexing, broadcasting, vectorized normalization
- **Pandas** — DataFrame operations, CSV I/O, `fillna`, `groupby`, `merge`, `apply`
- **Matplotlib** — histogram, scatter plot, bar chart, training loss line chart
- **PyTorch** — tensors, `nn.Sequential`, `BCELoss`, `Adam`, training loop, model save/load

---

## Setup

### 1. Clone or download the notebooks

Open both notebooks in [Google Colab](https://colab.research.google.com).

### 2. Get the Kaggle dataset

Download `diabetes.csv` from:
[kaggle.com/datasets/uciml/pima-indians-diabetes-database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

Upload it to Colab using the folder icon in the left sidebar when prompted.

### 3. Mount Google Drive

Both notebooks require Google Drive mounted to persist the saved model across sessions:

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## Running the Project

### Step 1 — Run the main training notebook (`diabetes_pipeline.ipynb`)

Run all cells top to bottom. The last cell saves the trained model to Google Drive:

```python
torch.save({
    'model_state': model.state_dict(),
    'X_mean': torch.tensor(X_mean),
    'X_std' : torch.tensor(X_std)
}, '/content/drive/MyDrive/diabetes_model.pth')
```

### Step 2 — Run the inference notebook (`test_on_real_data.ipynb`)

Paste and run the inference cell. It will:

1. Mount Google Drive and load the saved model
2. Prompt you to upload `diabetes.csv`
3. Fix hidden zero values in medical columns
4. Normalize using the original training statistics
5. Run predictions and print accuracy + confusion matrix

---

## Model Architecture

```
Input (5 features) → Linear(5→16) → ReLU → Linear(16→8) → ReLU → Linear(8→1) → Sigmoid
```

**Features used:** Age, BMI, Glucose, BloodPressure, Insulin

**Output:** Probability of diabetes (threshold 0.5)

**Loss function:** Binary Cross Entropy (`nn.BCELoss`)

**Optimizer:** Adam (lr=0.001)

---

## Important Design Decision — Normalization

The model was trained on synthetic data normalized with synthetic statistics (`X_mean`, `X_std`). When testing on the real Kaggle dataset, the **same training statistics** are applied — not the real dataset's own statistics. This is intentional and correct:

```python
# Right — use saved training stats
X_real_norm = (X_real - X_mean) / X_std

# Wrong — would put data on a different scale than what the model learned
X_real_norm = (X_real - X_real.mean()) / X_real.std()
```

---

## Known Dataset Issue

The Pima Indians dataset encodes missing values as `0` in medical columns. These are replaced with the column median before inference:

```python
zero_cols = ['Glucose', 'BloodPressure', 'Insulin', 'BMI']
df[zero_cols] = df[zero_cols].replace(0, np.nan)
for col in zero_cols:
    df[col].fillna(df[col].median(), inplace=True)
```

---

## Expected Results

Since the model was trained on synthetic data (not the real distribution), accuracy on real data will be lower than on synthetic test splits. This is expected and is itself a learning outcome — it demonstrates the impact of distribution shift between training and deployment data.

| Metric | Typical range |
|---|---|
| Accuracy | 60–72% |
| Patients tested | 768 |

---

## Concepts Covered

**NumPy**
- Array creation with `np.random`
- Boolean indexing and masking
- Reshaping and broadcasting
- Vectorized normalization (no loops)

**Pandas**
- Building DataFrames from arrays
- CSV read/write
- `fillna`, `isnull`, outlier filtering
- `apply` for custom transformations
- `groupby` + `agg`
- `merge` for joining tables
- `pd.cut` for binning

**Matplotlib**
- Histogram with overlapping distributions
- Scatter plot with color-coded classes
- Grouped bar chart
- Line chart for training loss
- Multi-panel layout with `plt.subplots`
- Saving figures with `savefig`

**PyTorch**
- Creating tensors from NumPy arrays
- `nn.Sequential` model definition
- Forward pass, loss computation, backpropagation
- `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()`
- `model.eval()` + `torch.no_grad()` for inference
- Saving and loading model checkpoints with `torch.save` / `torch.load`

---

## Troubleshooting

**`FileNotFoundError: diabetes_model.pth`**
Colab's local storage resets between sessions. Always save to and load from Google Drive (`/content/drive/MyDrive/`).

**`UnpicklingError` on model load**
The file was likely corrupted mid-save. Delete the file, re-run the training notebook, and save again. Convert NumPy arrays to tensors before saving:
```python
'X_mean': torch.tensor(X_mean),
'X_std' : torch.tensor(X_std)
```
Then load them back with `.numpy()`:
```python
X_mean = checkpoint['X_mean'].numpy()
X_std  = checkpoint['X_std'].numpy()
```

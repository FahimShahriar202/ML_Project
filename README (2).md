# Adult Income Prediction

A machine learning project that predicts whether a person earns **more than $50K per year** based on demographic and employment attributes from the Adult (Census Income) dataset. Four models are trained and compared: **K-Nearest Neighbors, Decision Tree, Logistic Regression, and a Neural Network**.

> **Author:** Fahim Shahriar Ahmed

---

## Table of Contents
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Models Used](#models-used)
- [Results](#results)
- [Key Observations](#key-observations)
- [Limitations and Future Work](#limitations-and-future-work)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)

---

## Dataset

The project uses the **Adult Income dataset** (UCI Machine Learning Repository / Census data).

| Property | Value |
|---|---|
| Rows | 48,842 |
| Columns | 15 (14 features + 1 target) |
| Problem type | Binary classification |
| Target (`target`) | `<=50K` (class 0) or `>50K` (class 1) |

**Features:**

| Type | Columns |
|---|---|
| Numerical | `Age`, `Final Weight`, `Education Number of Years`, `Capital-gain`, `Capital-loss`, `Hours-per-week` |
| Categorical | `Workclass`, `Education`, `Marital-status`, `Occupation`, `Relationship`, `Race`, `Sex`, `Native-country` |

**Class distribution:** about 76% earn `<=50K` and about 24% earn `>50K`, so the dataset is moderately imbalanced.

---

## Project Workflow

1. **Load the data** and inspect its shape, columns, and data types.
2. **Handle missing values**
   - Missing values were stored as `"?"`, so they were converted to `NaN`.
   - Three columns had missing values: `Workclass` (2,799), `Occupation` (2,809), and `Native-country` (857).
   - They were filled with the **mode** (most frequent value) of each column.
3. **Encode categorical features** into numbers using `LabelEncoder`.
4. **Correlation analysis** with a heatmap. No feature pairs had a correlation above 0.85, so no features were dropped.
5. **Train/test split:** 70% training and 30% testing, with `stratify=y` so both sets keep the same class ratio and `random_state=42` for reproducibility.
   - Train set: 34,189 rows
   - Test set: 14,653 rows
6. **Train four models** and make predictions.
7. **Compare the models** using accuracy, precision, recall, confusion matrices, and ROC curves with AUC.

---

## Models Used

| Model | Settings |
|---|---|
| K-Nearest Neighbors | default (`k = 5`) |
| Decision Tree | `random_state=42` |
| Logistic Regression | `max_iter=1000` |
| Neural Network (Keras) | Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(1, Sigmoid); Adam optimizer, binary cross-entropy loss, 10 epochs, batch size 32 |

---

## Results

Performance on the test set (14,653 rows). Precision, recall, and F1 below are for the **>50K class**, which is the harder and more important class to predict.

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| KNN | 77.6% | 0.55 | 0.32 | 0.41 |
| **Decision Tree** | **81.8%** | 0.62 | **0.63** | **0.62** |
| Logistic Regression | 81.0% | **0.67** | 0.40 | 0.50 |
| Neural Network | see note below | - | - | - |

> **Note on the Neural Network:** it was trained on unscaled features, and its training accuracy stayed around 69% with a very high loss (see [Limitations](#limitations-and-future-work)). Its test metrics and AUC are shown as plots in the notebook (bar charts, confusion matrix, and ROC curve).

The notebook also includes:
- Accuracy comparison bar chart
- Precision vs. recall comparison chart
- Confusion matrix for every model
- ROC curves with AUC scores for all four models

---

## Key Observations

- The **Decision Tree** gave the best overall balance, with the highest accuracy (81.8%) and by far the best recall (0.63) for high earners.
- **Logistic Regression** had the highest precision (0.67), but it misses most high earners (recall 0.40).
- **KNN** performed the worst. It is a distance-based model and was trained on unscaled features, which hurts it.
- Accuracy alone is misleading here: always predicting `<=50K` would already give about 76% accuracy. Recall and F1 for the `>50K` class give a more honest picture.

---

## Limitations and Future Work

- **Feature scaling was not applied.** `StandardScaler` is imported but not used. Scaling is important for KNN, Logistic Regression (which also showed a convergence warning), and especially the Neural Network, whose loss was in the hundreds. Adding scaling should noticeably improve these models.
- **Label encoding on nominal features** (such as `Occupation` or `Race`) gives the categories an artificial order. **One-hot encoding** would be more appropriate for linear models, KNN, and neural networks.
- **Missing values were imputed before the train/test split.** Computing the mode on the training set only would avoid a small amount of data leakage.
- **Class imbalance** (76% / 24%) was not addressed. Options include class weights, SMOTE, or threshold tuning.
- **Try stronger models** such as Random Forest, Gradient Boosting, or XGBoost, and tune hyperparameters with cross-validation (for example `GridSearchCV`, or limiting the Decision Tree depth to reduce overfitting).
- Use a **validation set / early stopping** and add Dropout or Batch Normalization to the neural network.

---

## How to Run

**Option 1: Google Colab**
1. Upload the notebook and the dataset (`adult income dataset.csv`) to Colab.
2. Make sure the file path in the notebook matches where you uploaded the CSV (the notebook uses `/content/sample_data/adult income dataset.csv`).
3. Click **Runtime -> Run all**.

**Option 2: Run locally**
```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# 2. Install the required libraries
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow jupyter

# 3. Launch the notebook
jupyter notebook AdultIncomePredictionProject.ipynb
```

> Update the CSV path in the first cell to point to the dataset file on your computer.

---

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib, Seaborn

# BreastCancerDemo
This project demonstrates a very basic Machine Learning Classification Task using Python and Scikit-learn.  The program uses the Breast Cancer Dataset to train a model that can classify whether a tumor is malignant or benign.
This project demonstrates a basic Machine Learning classification task using the Breast Cancer dataset from Scikit-learn.

The project includes:
- Data visualization
- Logistic Regression
- Confusion Matrix
- Accuracy
- Precision
- Recall
- AUC Score
## Requirements

Install required libraries using:

```bash
pip install pandas matplotlib scikit-learn
```
# Libraries Used

## Python

```python
import pandas as pd
import matplotlib.pyplot as plt
```

- `pandas` is used to work with tabular data (rows and columns).

- `matplotlib` is used to visualize data using graphs.
# Importing Libraries

```python
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)
```

This section imports all required libraries.

- `pandas` → Used for handling datasets in table format.
- `matplotlib` → Used for data visualization.
- `load_breast_cancer` → Loads the Breast Cancer dataset.
- `train_test_split` → Splits dataset into training and testing data.
- `LogisticRegression` → Machine Learning classification algorithm.
- `metrics` → Used to evaluate model performance.

---

# Loading the Dataset

```python
data = load_breast_cancer()
```

This line loads the Breast Cancer dataset into the variable `data`.

The dataset contains:
- Input features
- Output labels/classes

---

# Creating DataFrame

```python
df = pd.DataFrame(data.data, columns=data.feature_names)
```

This converts the dataset into a table format using Pandas DataFrame.

Each column represents a feature of the dataset.

Example features:
- mean radius
- mean texture
- mean area

---

# Adding Target Column

```python
df["target"] = data.target
```

This adds the target/output column.

Target values:
- `0` → Malignant
- `1` → Benign

---

# Displaying First 5 Rows

```python
print("First 5 Rows of Dataset:")
print(df.head())
```

`head()` displays the first 5 rows of the dataset.

This helps us understand:
- Dataset structure
- Features
- Labels

---

# Features and Labels

```python
X = data.data
y = data.target
```

- `X` contains input features.
- `y` contains output labels/classes.

In Machine Learning:
- Features = Inputs
- Labels = Outputs

---

# Visualizing the Dataset

```python
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Breast Cancer Dataset")
plt.show()
```

This creates a scatter plot of the dataset.

- `X[:, 0]` → First feature
- `X[:, 1]` → Second feature
- `c=y` → Colors points according to class labels

`show()` displays the graph.

---

# Splitting the Dataset

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

This splits the dataset into:
- Training Data (80%)
- Testing Data (20%)

Parameters:
- `test_size=0.2`
  - 20% data used for testing
- `random_state=42`
  - Ensures same random split every time

---

# Creating the Model

```python
model = LogisticRegression(max_iter=5000)
```

This creates a Logistic Regression model.

`max_iter=5000` increases the number of training iterations.

---

# Training the Model

```python
model.fit(X_train, y_train)
```

This trains the model using training data.

The model learns patterns from the dataset.

---

# Making Predictions

```python
y_pred = model.predict(X_test)
```

This predicts class labels for testing data.

---

# Probability Prediction

```python
y_prob = model.predict_proba(X_test)[:, 1]
```

This predicts probability values for each class.

Used for calculating AUC score.

---

# Confusion Matrix

```python
cm = confusion_matrix(y_test, y_pred)
```

Confusion Matrix shows:
- Correct predictions
- Incorrect predictions

It helps analyze model performance in detail.

---

# Accuracy

```python
accuracy = accuracy_score(y_test, y_pred)
```

Accuracy measures overall correctness of the model.

Formula:

Accuracy = (TP + TN) / (TP + TN + FP + FN)

---

# Precision

```python
precision = precision_score(y_test, y_pred)
```

Precision measures:

Out of all predicted positive cases,
how many were actually positive.

Formula:

Precision = TP / (TP + FP)

---

# Recall

```python
recall = recall_score(y_test, y_pred)
```

Recall measures:

Out of all actual positive cases,
how many were correctly identified.

Formula:

Recall = TP / (TP + FN)

---

# AUC Score

```python
auc = roc_auc_score(y_test, y_prob)
```

AUC measures the model's ability to distinguish between classes.

AUC Range:
- `1.0` → Perfect model
- `0.5` → Random guessing

Higher AUC means better performance.

---

# Printing Results

```python
print("\nConfusion Matrix:")
print(cm)

print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("AUC Score:", auc)
```

This section displays:
- Confusion Matrix
- Accuracy
- Precision
- Recall
- AUC Score

---

# Machine Learning Workflow

1. Load Dataset
2. Visualize Dataset
3. Split Dataset
4. Create Model
5. Train Model
6. Make Predictions
7. Evaluate Performance

---

# Concepts Learned

After completing this project, beginners will understand:
- Dataset
- Features and labels
- Classification
- Training and testing
- Logistic Regression
- Predictions
- Evaluation metrics
- Data visualization
- Basic Machine Learning workflow

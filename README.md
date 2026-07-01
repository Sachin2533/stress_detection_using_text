# Stress Detection Using Text

## Project Overview

This project detects whether a given text indicates **Stress** or **No Stress** using Natural Language Processing and Machine Learning techniques.

The dataset contains text messages labeled as either stressed or non-stressed. The text data is cleaned, converted into numerical features using `CountVectorizer`, and then trained on multiple machine learning classification algorithms. The final models are evaluated using accuracy score, classification report, and confusion matrix.

---

## Objective

The main objective of this project is to build a text classification model that can predict whether a user's input text reflects stress or no stress.

---

## Dataset

The dataset used in this project is:

```text
preprocessedNPShuffled.csv
```

The dataset contains two main columns:

| Column  | Description                                                 |
| ------- | ----------------------------------------------------------- |
| `text`  | Text message or sentence                                    |
| `label` | Target label where `0` means No Stress and `1` means Stress |

After loading the dataset, the columns are renamed as:

```python
data.columns = ['text', 'label']
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* NLTK
* Scikit-learn

---

## Project Workflow

```text
Import Dataset
      |
      v
Rename Columns
      |
      v
Text Cleaning
      |
      v
Label Mapping
      |
      v
Text Vectorization
      |
      v
Train-Test Split
      |
      v
Model Training
      |
      v
Model Evaluation
      |
      v
User Text Prediction
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/stress-detection-using-text.git
cd stress-detection-using-text
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

For macOS/Linux:

```bash
source venv/bin/activate
```

### 4. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn
```

---

## Required NLTK Data

This project uses English stopwords from NLTK.

Run the following command once before running the project:

```python
import nltk
nltk.download('stopwords')
```

---

## Importing Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import string

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
```

---

## Data Loading

```python
data = pd.read_csv("preprocessedNPShuffled.csv")
data.columns = ['text', 'label']
data.head()
```

The dataset contains **2303 rows** and **2 columns**.

Example records:

| text                                   | label |
| -------------------------------------- | ----- |
| Thank you happy.1                      | 0     |
| That would be a great trick happy.3    | 0     |
| unhappy every time laughing my ass off | 1     |
| have a blast okay. love you!           | 1     |
| Enjoy happy                            | 0     |

---

## Text Cleaning

Text data is cleaned before training the machine learning models.

The cleaning process includes:

* Converting text to lowercase
* Removing URLs
* Removing HTML tags
* Removing punctuation
* Removing newline characters
* Removing words containing numbers
* Removing stopwords

```python
stopword = set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    return text

data["text"] = data["text"].apply(clean)
```

---

## Label Mapping

The numeric labels are converted into readable class names.

```python
data["label"] = data["label"].map({
    0: "No Stress",
    1: "Stress"
})
```

Final label format:

| Numeric Label | Class Name |
| ------------- | ---------- |
| 0             | No Stress  |
| 1             | Stress     |

---

## Feature Extraction

The text data is converted into numerical form using `CountVectorizer`.

```python
from sklearn.feature_extraction.text import CountVectorizer

x = np.array(data["text"])
y = np.array(data["label"])

cv = CountVectorizer()
X = cv.fit_transform(x)
```

`CountVectorizer` converts text into a matrix of token counts, which can be used by machine learning models.

---

## Train-Test Split

The dataset is split into training and testing sets.

```python
from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.33, random_state=42
)
```

* Training data: 67%
* Testing data: 33%

---

## Machine Learning Models Used

The following machine learning models are trained and compared:

1. Bernoulli Naive Bayes
2. Logistic Regression
3. Decision Tree Classifier
4. K-Nearest Neighbors Classifier
5. Support Vector Classifier
6. Random Forest Classifier

---

## Model Training and Evaluation

### 1. Bernoulli Naive Bayes

```python
from sklearn.naive_bayes import BernoulliNB

model = BernoulliNB()
model.fit(xtrain, ytrain)
```

Accuracy:

```text
88.94%
```

Classification report:

```text
              precision    recall  f1-score   support

   No Stress       0.91      0.89      0.90       423
      Stress       0.87      0.89      0.88       337

    accuracy                           0.89       760
   macro avg       0.89      0.89      0.89       760
weighted avg       0.89      0.89      0.89       760
```

Confusion matrix:

```text
[[377,  46],
 [ 38, 299]]
```

---

### 2. Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

model2 = LogisticRegression()
model2.fit(xtrain, ytrain)
```

Accuracy:

```text
91.44%
```

Confusion matrix:

```text
[[415,   8],
 [ 57, 280]]
```

---

### 3. Decision Tree Classifier

```python
from sklearn.tree import DecisionTreeClassifier

model3 = DecisionTreeClassifier()
model3.fit(xtrain, ytrain)
```

Accuracy:

```text
92.10%
```

Confusion matrix:

```text
[[410,  13],
 [ 47, 290]]
```

---

### 4. K-Nearest Neighbors Classifier

```python
from sklearn.neighbors import KNeighborsClassifier

model4 = KNeighborsClassifier()
model4.fit(xtrain, ytrain)
```

Accuracy:

```text
88.81%
```

Confusion matrix:

```text
[[405,  18],
 [ 67, 270]]
```

---

### 5. Support Vector Classifier

```python
from sklearn.svm import SVC

model5 = SVC(C=2)
model5.fit(xtrain, ytrain)
```

Accuracy:

```text
91.71%
```

Confusion matrix:

```text
[[419,   4],
 [ 59, 278]]
```

---

### 6. Random Forest Classifier

```python
from sklearn.ensemble import RandomForestClassifier

model6 = RandomForestClassifier()
model6.fit(xtrain, ytrain)
```

Accuracy:

```text
91.71%
```

Confusion matrix:

```text
[[412,  11],
 [ 52, 285]]
```

---

## Model Comparison

| Algorithm           | Accuracy |
| ------------------- | -------: |
| Naive Bayes         |   88.94% |
| Logistic Regression |   91.44% |
| Decision Tree       |   92.10% |
| KNN                 |   88.81% |
| SVC                 |   91.71% |
| Random Forest       |   91.71% |

---

## Best Performing Model

The **Decision Tree Classifier** achieved the highest accuracy of:

```text
92.10%
```

Therefore, it performed best among the tested models.

---

## Confusion Matrix Visualization

The confusion matrix is visualized using Seaborn heatmap.

```python
plt.figure(figsize=(10,10))
sns.heatmap(confusion_matrix(ytest, pred_decs), annot=True, cmap='Blues')
plt.xlabel('prediction')
plt.ylabel('truth')
plt.show()
```

---

## Predicting Stress From User Input

The project allows users to enter custom text and predict whether it indicates stress or no stress.

Example:

```python
user = input("Enter a Text: ")
data = cv.transform([user]).toarray()
output = model3.predict(data)
print(output)
```

Sample output:

```text
['No Stress']
```

---

## Project Structure

```text
stress-detection-using-text/
│
├── preprocessedNPShuffled.csv
├── stress_detection.ipynb
├── README.md
└── requirements.txt
```

---

## Requirements File

Create a `requirements.txt` file with the following dependencies:

```text
pandas
numpy
matplotlib
seaborn
nltk
scikit-learn
```

---

## How to Run the Project

### Run the Jupyter Notebook

```bash
jupyter notebook
```

Open the notebook file and run all cells step by step.

### Run as a Python Script

If the code is saved as a Python file, run:

```bash
python stress_detection.py
```

---

## Key Findings

* Text preprocessing improves model performance.
* `CountVectorizer` is used to convert text into numerical features.
* Decision Tree Classifier achieved the highest accuracy.
* Logistic Regression, SVC, and Random Forest also performed well.
* The model can classify user input text into `Stress` or `No Stress`.

---

## Future Improvements

* Use TF-IDF Vectorizer instead of CountVectorizer.
* Apply stemming or lemmatization.
* Use deep learning models such as LSTM, GRU, or BERT.
* Save the best model using Joblib or Pickle.
* Build a web application using Streamlit or Flask.
* Add real-time text input and prediction interface.
* Improve dataset quality and class balance.
* Add more evaluation metrics and visualizations.

---

## Author

**Sachin Thakur**

---

## License

This project is open-source and available for learning and educational purposes.

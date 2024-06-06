{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## ***Stress Detection Using Text***\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "***Importng Libraries And Datasset***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "data = pd.read_csv(\"preprocessedNPShuffled.csv\")\n",
    "\n",
    "data.columns = ['text', 'label']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>text</th>\n",
       "      <th>label</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Thank you happy.1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>That would be a great trick happy.3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>unhappy  every time laughing my ass off</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>have a blast okay. love you!</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Enjoy happy</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                      text  label\n",
       "0                        Thank you happy.1      0\n",
       "1      That would be a great trick happy.3      0\n",
       "2  unhappy  every time laughing my ass off      1\n",
       "3             have a blast okay. love you!      1\n",
       "4                              Enjoy happy      0"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## ***Checking the Dataset***\n",
    "(For Data wrangling)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>text</th>\n",
       "      <th>label</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Thank you happy.1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>That would be a great trick happy.3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>unhappy  every time laughing my ass off</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>have a blast okay. love you!</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Enjoy happy</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2298</th>\n",
       "      <td>does it have flip out scr</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2299</th>\n",
       "      <td>but happy birthday hoe! have a good day</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2300</th>\n",
       "      <td>instant message perfectly fine :D how about yo...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2301</th>\n",
       "      <td>I lost one import quizzz unhappy</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2302</th>\n",
       "      <td>keep smiling happy.1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>2303 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                   text  label\n",
       "0                                     Thank you happy.1      0\n",
       "1                   That would be a great trick happy.3      0\n",
       "2               unhappy  every time laughing my ass off      1\n",
       "3                          have a blast okay. love you!      1\n",
       "4                                           Enjoy happy      0\n",
       "...                                                 ...    ...\n",
       "2298                         does it have flip out scr       1\n",
       "2299            but happy birthday hoe! have a good day      0\n",
       "2300  instant message perfectly fine :D how about yo...      0\n",
       "2301                  I lost one import quizzz unhappy       1\n",
       "2302                               keep smiling happy.1      0\n",
       "\n",
       "[2303 rows x 2 columns]"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "import nltk\n",
    "import re\n",
    "# nltk.download('stopwords')\n",
    "# stemmer = nltk.SnowballStemmer(\"english\")\n",
    "from nltk.corpus import stopwords\n",
    "import string\n",
    "stopword=set(stopwords.words('english'))\n",
    "\n",
    "def clean(text):\n",
    "    text = str(text).lower()\n",
    "    text = re.sub('\\[.*?\\]', '', text)\n",
    "    text = re.sub('https?://\\S+|www\\.\\S+', '', text)\n",
    "    text = re.sub('<.*?>+', '', text)\n",
    "    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)\n",
    "    text = re.sub('\\n', '', text)\n",
    "    text = re.sub('\\w*\\d\\w*', '', text)\n",
    "    text = [word for word in text.split(' ') if word not in stopword]\n",
    "    text=\" \".join(text)\n",
    "    # text = [stemmer.stem(word) for word in text.split(' ')]\n",
    "    return text\n",
    "data[\"text\"] = data[\"text\"].apply(clean)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>text</th>\n",
       "      <th>label</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>thank</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>would great trick</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>unhappy  every time laughing ass</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>blast okay love</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>enjoy happy</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                               text  label\n",
       "0                            thank       0\n",
       "1                would great trick       0\n",
       "2  unhappy  every time laughing ass      1\n",
       "3                   blast okay love      1\n",
       "4                       enjoy happy      0"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>text</th>\n",
       "      <th>label</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>thank</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>would great trick</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>unhappy  every time laughing ass</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>blast okay love</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>enjoy happy</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2298</th>\n",
       "      <td>flip scr</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2299</th>\n",
       "      <td>happy birthday hoe good day</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2300</th>\n",
       "      <td>instant message perfectly fine fellow ken</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2301</th>\n",
       "      <td>lost one import quizzz unhappy</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2302</th>\n",
       "      <td>keep smiling</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>2303 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                                           text  label\n",
       "0                                        thank       0\n",
       "1                            would great trick       0\n",
       "2              unhappy  every time laughing ass      1\n",
       "3                               blast okay love      1\n",
       "4                                   enjoy happy      0\n",
       "...                                         ...    ...\n",
       "2298                                  flip scr       1\n",
       "2299                happy birthday hoe good day      0\n",
       "2300  instant message perfectly fine fellow ken      0\n",
       "2301            lost one import quizzz unhappy       1\n",
       "2302                              keep smiling       0\n",
       "\n",
       "[2303 rows x 2 columns]"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "***Now, Labeling the the Text, [Stress,No Stress]***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                               text      label\n",
      "0                            thank   No Stress\n",
      "1                would great trick   No Stress\n",
      "2  unhappy  every time laughing ass     Stress\n",
      "3                   blast okay love     Stress\n",
      "4                       enjoy happy  No Stress\n"
     ]
    }
   ],
   "source": [
    "data[\"label\"] = data[\"label\"].map({0: \"No Stress\", 1: \"Stress\"})\n",
    "data = data[[\"text\", \"label\"]]\n",
    "print(data.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "x = np.array(data[\"text\"])\n",
    "y = np.array(data[\"label\"])\n",
    "\n",
    "cv = CountVectorizer()\n",
    "X = cv.fit_transform(x)\n",
    "xtrain, xtest, ytrain, ytest = train_test_split(X, y,test_size=0.33,random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "BernoulliNB()"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.naive_bayes import BernoulliNB\n",
    "from sklearn import metrics\n",
    "model = BernoulliNB()\n",
    "model.fit(xtrain, ytrain)\n",
    "# print(\"Accuracy of Logistic Regression model is:\",\n",
    "# metrics.accuracy_score(ytest, ytrain)*100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['No Stress']\n"
     ]
    }
   ],
   "source": [
    "user = input(\"Enter a Text: \")\n",
    "data = cv.transform([user]).toarray()\n",
    "output = model.predict(data)\n",
    "print(output)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "88.94736842105263"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "scores = []\n",
    "scores.append(model.score(xtest,ytest))\n",
    "# scoresdf\n",
    "# dict = {'Algorithm':'Naive Bayes','Score':model.score(xtest,ytest)}\n",
    "# scoresdf  = scoresdf.append(dict, ignore_index = True)\n",
    "model.score(xtest,ytest)*100\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "   No Stress       0.91      0.89      0.90       423\n",
      "      Stress       0.87      0.89      0.88       337\n",
      "\n",
      "    accuracy                           0.89       760\n",
      "   macro avg       0.89      0.89      0.89       760\n",
      "weighted avg       0.89      0.89      0.89       760\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import classification_report\n",
    "pred_bern = model.predict(xtest)\n",
    "print(classification_report(ytest, pred_bern))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[377,  46],\n",
       "       [ 38, 299]], dtype=int64)"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.metrics import confusion_matrix\n",
    "confusion_matrix(ytest,pred_bern)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LogisticRegression()"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "model2 = LogisticRegression()\n",
    "model2.fit(xtrain,ytrain)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "91.44736842105263"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "scores.append(model2.score(xtest,ytest))\n",
    "dict = {'Algorithm':'LogisticRegression','Score':model2.score(xtest,ytest)}\n",
    "model2.score(xtest,ytest)*100\n",
    "# scoresdf  = scoresdf.append(dict, ignore_index = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['No Stress']\n"
     ]
    }
   ],
   "source": [
    "user = input(\"Enter a Text: \")\n",
    "data = cv.transform([user]).toarray()\n",
    "output = model2.predict(data)\n",
    "print(output)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[415,   8],\n",
       "       [ 57, 280]], dtype=int64)"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.metrics import confusion_matrix\n",
    "pred_logis = model2.predict(xtest)\n",
    "confusion_matrix(ytest,pred_logis)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DecisionTreeClassifier()"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.tree import DecisionTreeClassifier\n",
    "model3 = DecisionTreeClassifier()\n",
    "model3.fit(xtrain,ytrain)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "92.10526315789474"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "scores.append(model3.score(xtest,ytest))\n",
    "model3.score(xtest,ytest)*100"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['No Stress']\n"
     ]
    }
   ],
   "source": [
    "user = input(\"Enter a Text: \")\n",
    "data = cv.transform([user]).toarray()\n",
    "output = model3.predict(data)\n",
    "print(output)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[410,  13],\n",
       "       [ 47, 290]], dtype=int64)"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pred_decs = model3.predict(xtest)\n",
    "from sklearn.metrics import confusion_matrix\n",
    "confusion_matrix(ytest,pred_decs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(69.0, 0.5, 'truth')"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 720x720 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(10,10))\n",
    "sns.heatmap(confusion_matrix(ytest,pred_decs),annot=True,cmap='Blues')\n",
    "plt.xlabel('prediction')\n",
    "plt.ylabel('truth')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "KNeighborsClassifier()"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "model4= KNeighborsClassifier()\n",
    "model4.fit(xtrain,ytrain)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "88.81578947368422"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "scores.append(model4.score(xtest,ytest))\n",
    "model4.score(xtest,ytest)*100"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['No Stress']\n"
     ]
    }
   ],
   "source": [
    "user = input(\"Enter a Text: \")\n",
    "data = cv.transform([user]).toarray()\n",
    "output = model4.predict(data)\n",
    "print(output)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "pred_knn = model4.predict(xtest)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[405,  18],\n",
       "       [ 67, 270]], dtype=int64)"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.metrics import confusion_matrix\n",
    "confusion_matrix(ytest,pred_knn)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "SVC(C=2)"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.svm import SVC\n",
    "model5 = SVC(C=2)\n",
    "model5.fit(xtrain,ytrain)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "91.71052631578948"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model5.score(xtest,ytest)*100"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['No Stress']\n"
     ]
    }
   ],
   "source": [
    "user = input(\"Enter a Text: \")\n",
    "data = cv.transform([user]).toarray()\n",
    "output = model5.predict(data)\n",
    "print(output)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "pred_svm=model5.predict(xtest)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[419,   4],\n",
       "       [ 59, 278]], dtype=int64)"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.metrics import confusion_matrix\n",
    "confusion_matrix(ytest,pred_svm)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomForestClassifier()"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "model6 = RandomForestClassifier()\n",
    "model6.fit(xtrain,ytrain)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9171052631578948"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model6.score(xtest,ytest)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "pred_rand = model6.predict(xtest)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['No Stress']\n"
     ]
    }
   ],
   "source": [
    "user = input(\"Enter a Text: \")\n",
    "data = cv.transform([user]).toarray()\n",
    "output = model6.predict(data)\n",
    "check = output\n",
    "print(output)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[412,  11],\n",
       "       [ 52, 285]], dtype=int64)"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.metrics import confusion_matrix\n",
    "confusion_matrix(ytest,pred_rand)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [],
   "source": [
    "dict = {'Algo':['NaiveBayes','LogisticRegression','DecisionTree','KNN','SVC','RandomForest'],\n",
    "        'Score':[model.score(xtest,ytest),model2.score(xtest,ytest),model3.score(xtest,ytest),model4.score(xtest,ytest),\n",
    "                 model5.score(xtest,ytest),model6.score(xtest,ytest)]}\n",
    "scoresdf = pd.DataFrame.from_dict(dict)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Algo</th>\n",
       "      <th>Score</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>NaiveBayes</td>\n",
       "      <td>0.889474</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>LogisticRegression</td>\n",
       "      <td>0.914474</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>DecisionTree</td>\n",
       "      <td>0.921053</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>KNN</td>\n",
       "      <td>0.888158</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>SVC</td>\n",
       "      <td>0.917105</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>RandomForest</td>\n",
       "      <td>0.917105</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                 Algo     Score\n",
       "0          NaiveBayes  0.889474\n",
       "1  LogisticRegression  0.914474\n",
       "2        DecisionTree  0.921053\n",
       "3                 KNN  0.888158\n",
       "4                 SVC  0.917105\n",
       "5        RandomForest  0.917105"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "scoresdf"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  },
  "vscode": {
   "interpreter": {
    "hash": "1c697d69f927956c0f87158e03f309f8069fc71ae87cd648e9b5b7140a5374b5"
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

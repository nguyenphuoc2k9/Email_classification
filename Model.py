
import numpy as np
import pandas as pd
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('./data/2cls_spam_text_cls.csv')

messages = df['Message'].values.tolist()
labels = df['Category'].values.tolist()

def lowercase(text):
  return text.lower()
def punctuation(text):
  return text.translate(str.maketrans('', '', string.punctuation))
def tokenize(text):
  return nltk.word_tokenize(text)
def remove_stopwords(tokens):
  stop_words = nltk.corpus.stopwords.words('english')
  return [word for word in tokens if word not in stop_words]
def stemming(tokens):
  stemmer = nltk.stem.PorterStemmer()
  return [stemmer.stem(word) for word in tokens]
def preprocess(text):
  text = lowercase(text)
  text = punctuation(text)
  tokens = tokenize(text)
  tokens = remove_stopwords(tokens)
  tokens = stemming(tokens)
  return tokens
messages = [preprocess(message) for message in messages]

def create_dictionary(messages):
  dictionary = []
  for tokens in messages:
    for token in tokens:
      if token not in dictionary:
        dictionary.append(token)
  return dictionary
dictionary = create_dictionary(messages)

def create_feature(tokens,dictionary):
  features = np.zeros(len(dictionary))
  for token in tokens:
    if token in dictionary:
      features[dictionary.index(token)] +=1
  return features
X = [create_feature(message,dictionary) for message in messages]

le = LabelEncoder()
y = le.fit_transform(labels)

VAL_SIZE =0.2
TEST_SIZE = 0.125
SEED = 0

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=VAL_SIZE,random_state=SEED,shuffle = True)

X_train,X_test,y_train,y_test = train_test_split(X_train,y_train,test_size=TEST_SIZE,random_state=SEED,shuffle = True)

model = GaussianNB()

model = model.fit(X_train,y_train)

def predict(text,model,dictionary):
  processed_text = preprocess(text)
  features = create_feature(processed_text,dictionary)
  features = np.array(features).reshape(1,-1)
  prediction = model.predict(features)
  prediction_cls = le.inverse_transform(prediction)
  return prediction_cls[0]
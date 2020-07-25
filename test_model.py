import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import torch
df = pd.read_csv('./IMDB Dataset.csv',delimiter=',',encoding='latin-1')
df.head()

x = df.review
y = df.sentiment
le = LabelEncoder()
y = le.fit_transform(y)
max_words = 3000
max_len = 200
token = Tokenizer(num_words=max_words)
token.fit_on_texts(x)
seq = token.texts_to_sequences(x)
matrix = sequence.pad_sequences(seq,maxlen=max_len)
xtrain,xtest,ytrain,ytest = train_test_split(matrix,y,test_size=0.3)
embedding_dim=50
model=Sequential()

for i in range(15):
  print(" FINDING ACCURACY FOR THE MODEL WHICH HAS EPOCHS: ",i)
  model_load_name = i
  model = torch.load(str(i))  
  loss, acc = model.evaluate(xtrain, ytrain)
  print("Training Accuracy: ", acc,loss)
  loss, acc = model.evaluate(xtest, ytest)
  print("Test Accuracy: ", acc,loss)
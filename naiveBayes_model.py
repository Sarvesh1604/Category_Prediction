import string 
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import csv

data_train = pd.read_csv('traindata.csv')
catagory_train = data_train.iloc[:, 0].values
sent_train = data_train.iloc[:, 1].values

catagory = dict()
for label in catagory_train:
  catagory[label] = catagory.get(label, 0) +1
  

#Prior distribution of labels
total = catagory_train.shape[0]
prob_prior = dict()

for key in catagory.keys():
  prob_prior[key] = catagory[key]/total


table = str.maketrans('','', string.punctuation)
i = 0
vocab = dict()
vocab_whole = []
for key in catagory.keys():
  vocab[key] = []
  
#creating a vocabulary with every word of each topic(vocab_whole), 
#and four other vocabularies for each topic(vocab):
for sentence in sent_train:
  words = sentence.split()
  words = [word.translate(table) for word in words]
  words = [word.lower() for word in words]

  words_final = []
  stop_words = stopwords.words('english')
  for word in words:
    if word not in stop_words:
      words_final.append(word)

  key = catagory_train[i]
  for word in words_final:
    vocab.get(key).append(word)
    vocab_whole.append(word)
  i += 1

#calculating class conditional probability for each word:
prob_class_cond = {'science': {}, 'sports': {}, 'business': {}, 'covid': {}}

for word in vocab_whole:
  for key in catagory_train:
    count = 0
    for found_word in vocab.get(key):
      if found_word == word:
        count += 1
    prob_class_cond[key][word] = count/len(vocab.get(key))
    if prob_class_cond[key][word] == 0:
      prob_class_cond[key][word] = 0.0001
      
      
#Posterior distribution for each test sentence and
#catagory prediction
data_test = pd.read_csv('testdata.csv')
sent_test = data_test.iloc[:, 1].values

predicted_label = []
i = 0
for sentence in sent_test:
  words = sentence.split()
  words = [word.translate(table) for word in words]
  words = [word.lower() for word in words]

  words_final = []
  stop_words = stopwords.words('english')
  for word in words:
    if word not in stop_words:
      words_final.append(word)

  prob_post = {'science': prob_prior['science'], 'sports': prob_prior['sports'], 'business': prob_prior['business'], 'covid': prob_prior['covid']}

  for word in words_final:
    for key in prob_post.keys():
      try:
        prob_post[key] *= prob_class_cond[key][word] 
      except:
        prob_post[key] *= 0.0001

  max = 0
  
  for key in prob_post.keys():
    if prob_post[key] > max:
      max = prob_post[key]
      predicted_label.append(key)
  
with open('predictions.csv', 'w', newline='') as file:
  writer = csv.writer(file)
  writer.writerow(['text', 'category'])
  i = 0
  for sentence in data_test['text']:
    writer.writerow([sentence, predicted_label[i]])
    i += 1
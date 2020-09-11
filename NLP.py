import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#csv File delimiter is a comma but reviews have comma already in them. Thus we use a tsv file which uses tab as a delimiter.
dataset = pd.read_csv('Twitter_Sentiments.tsv', delimiter = '\t', quoting = 3) #quoting=3 Ignores double quotes.
rows = len(dataset)

#Cleaning the texts
import re
import nltk #Remove Unecessary words
#nltk.download('stopwords') - Location downloaded is in corpus folder
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
stemObj = PorterStemmer()

corpus = [] #Clean Text list
for i in range(rows):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) #Substitue all char except letter with a white space
    review = review.lower()
    review = review.split()
    review = [stemObj.stem(word) for word in review if not stemObj.stem(word) in set(stopwords.words('english'))] #set increases efficiency for bigger texts
    review = ' '.join(review)
    corpus.append(review)
    
#Creating Bag of Words Model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) #Can do everything we did above but manually is fast  
X = cv.fit_transform(corpus).toarray()
Y = dataset.iloc[:, 1].values


#Naive Bayes Template
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)


#Define New Classifier for the training set below
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)


Y_pred = classifier.predict(X_test)

#Making the confusion Matrix
from sklearn.metrics import confusion_matrix
ConfMat = confusion_matrix(Y_test, Y_pred)
print(ConfMat)

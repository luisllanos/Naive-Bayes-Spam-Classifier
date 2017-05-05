


# split into training and testing sets
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Dataset from - https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
df = pd.read_table('smsspamcollection/SMSSpamCollection',
                   sep='\t', 
                   header=None, 
                   names=['label', 'sms_message'])

# Output printing out first 5 columns
df.head()

df['label'] = df.label.map({'ham':0, 'spam':1})
print(df.shape)
df.head() # returns (rows, columns)


X_train, X_test, y_train, y_test = train_test_split(df['sms_message'], 
                                                    df['label'], 
                                                    random_state=1)

print('Number of rows in the total set: {}'.format(df.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))

count_vector = CountVectorizer()
print(count_vector)

# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)

# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
testing_data = count_vector.transform(X_test)

'''
we will be using the multinomial Naive Bayes implementation. 
This particular classifier is suitable for classification 
with discrete features (such as in our case, word counts 
for text classification). It takes in integer word counts as its input. 
On the other hand Gaussian Naive Bayes is better suited for continuous 
data as it assumes that the input data has a Gaussian(normal) distribution.'''
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)


# Naive Bayes implementation using scikit-learn
predictions = naive_bayes.predict(testing_data)

#Evaluating our model
print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
print('Precision score: ', format(precision_score(y_test, predictions)))
print('Recall score: ', format(recall_score(y_test, predictions)))
print('F1 score: ', format(f1_score(y_test, predictions)))

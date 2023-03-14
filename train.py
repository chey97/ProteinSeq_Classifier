
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from data_filter import data
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from tqdm import tqdm


# ------ Train Test Split ------------------------------------------------------------------------

# Split Data
X_train, X_test, y_train, y_test = train_test_split(data['sequence'], data['classification'], test_size = 0.2, random_state = 42)


# After splitting the data, it's important to utilize the CountVectorizer to create a dictionary composed from the training dataset. 
vect = CountVectorizer( analyzer = 'char_wb', ngram_range = (4 , 4) )

# Fit and Transform CountVectorizer
vect.fit(X_train)
X_train_df = vect.transform(X_train)
X_test_df = vect.transform(X_test)

# Print a few of the features
print(vect.get_feature_names_out()[-20:])

# ------ Machine Learning Models ------

# Make a prediction dictionary to store accuracies
prediction = dict()

# set the number of epochs
epochs = 2

# fit the model for each epoch and update the progress bar
for epoch in range(epochs):
    model = MultinomialNB()
    model.fit(X_train_df, y_train)
    NB_pred = model.predict(X_test_df)
    acc = accuracy_score(NB_pred, y_test)
    prediction["MultinomialNB"] = acc
    print (acc)
    
'''    # update progress bar
    tqdm.write(f"Epoch {epoch+1}/{epochs} - Accuracy: {acc}") 
    
# -------------------------------------------------------------------

# Adaptive boosting

model = AdaBoostClassifier()
model.fit(X_train_df,y_train)
ADA_pred = model.predict(X_test_df)
prediction["Adaboost"] = accuracy_score(ADA_pred , y_test)
print(prediction["Adaboost"]) '''







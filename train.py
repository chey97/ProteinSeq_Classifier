
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from data_filter import data


# ------ Train Test Split ------------------------------------------------------------------------

# Split Data
X_train, X_test, y_train, y_test = train_test_split(data['sequence'], data['classification'], test_size = 0.2, random_state = 1)


# After splitting the data, it's important to utilize the CountVectorizer to create a dictionary composed from the training dataset. 
vect = CountVectorizer( analyzer = 'char_wb', ngram_range = (4 , 4) )

# Fit and Transform CountVectorizer
vect.fit(X_train)
X_train_df = vect.transform(X_train)
X_test_df = vect.transform(X_test)

# Print a few of the features
print(vect.get_feature_names())
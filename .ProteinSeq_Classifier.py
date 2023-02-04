'''
- This program was build for academic purpose - ProteinSeq - Protein Sequence Classifier Copyright (C) 2023  Chethiya Galkaduwa
'''

# importing libraries

import pandas as pd 
import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns 
# from sklearn.feature_extraction.text import CounterVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# importing the datasets

df_char = pd.read_csv('data/pdb_data_no_dups.csv')
df_seq = pd.read_csv('data/pdb_data_seq.csv')
print(df_seq.head())

# ------- Filter and Process Data -----------------------------------------------------------------------------------------------------------------
# With the data loaded into two seperate pandas dataframes, a filter, project, and a join must be performed to get the data together.
# Filtering proteins only.
protein_char = df_char[df_char.macromoleculeType == 'Protein']
protein_seq = df_seq[df_seq.macromoleculeType == 'Protein']

# Selecting only the necessary variables to join 
protein_char = protein_char[['structureId','classification']]
protein_seq = protein_seq[['structureId','sequence']]

# print(protein_seq.head()) # display the first five rows of the protein_seq dataframe.
# print(protein_char.head())

model_f = protein_char.set_index('structureId').join(protein_seq.set_index('structureId')) 
# Joining the two data frames into one with 346,325 proteins
#print(model_f.head())
print('Number of rows in the joined dataset: ', model_f.shape)

# Removing the NA columns(missingness of values in the columns)
#check NA counts
model_f.isnull().sum()

# Dropping the rows with missing values (NULL values)
model_f = model_f.dropna()
#print(model_f)
print('Number of remainning rows: ', model_f.shape)

# looking at the types of family groups that clasification can be sorted into.

counts = model_f.classification.value_counts() 
# count the occurrences of each unique value in the classification column of the model_f dataframe and returns a series object that contains the counts of unique values in descending order.
print(counts)

df = pd.DataFrame({'classification' : counts})
print(df)
df.to_csv('Classification_family_groups.csv')

# --- ploting the counts ---- visualize the distribution of the number of records for each family type

if not os.path.exists("plots"):
    os.makedirs("plots")

plt.figure()
sns.displot(data=df, x='classification', kind='kde', color = 'blue')
plt.title('Count Distribution for Family Types')
plt.ylabel('% of records')
plt.savefig("plots/Count Distribution for Family Types.png")
# plt.show()

# Get classification types where counts are over 1000
types = np.asarray(counts[(counts > 1000)].index)

# Filter dataset's records for classification types > 1000
data = model_f[model_f.classification.isin(types)] #  The 'isin' method checks if each value in the column is contained in the list types

print(types)
print('%d is the number of records in the final filtered dataset' %data.shape[0])

# ------ Train Test Split ------------------------------------------------------------------------

# Splot Data
X_train, X_test, y_train, y_test = train_test_split(data['sequence'], data['classification'], test_size = 0.2, random_state = 1)


# After splitting the data, it's important to utilize the CountVectorizer to create a dictionary composed from the training dataset. 


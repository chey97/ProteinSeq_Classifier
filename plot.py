'''
- This program was build for academic purpose - ProteinSeq - Protein Sequence Classifier Copyright (C) 2023  Chethiya Galkaduwa
'''
from train import y_test, NB_pred
from data_filter import *


# Visualize Metrics | Plotting the confussion metrics

if not os.path.exists("plots"):
    os.makedirs("plots")

# Plot confusion matrix
conf_mat = confusion_matrix(y_test, NB_pred, labels = types)

#Normalize confusion_matrix
conf_mat = conf_mat.astype('float')/ conf_mat.sum(axis=1)[:, np.newaxis]

# Plot Heat Map
fig , ax = plt.subplots()
fig.set_size_inches(13, 8)
sns.heatmap(conf_mat)
plt.title('Heat Map')
plt.ylabel('Types')
plt.savefig("plots/Heat Map")


# The confusion matrix shows label index 3 being misclassified as index 38 quite a bit. Based on the names listed below, it makes sense for these two to be confused.
print(types[3])
print(types[38])


#Print F1 score metrics
report = (classification_report(y_test, NB_pred, target_names = types))
df = pd.DataFrame(report).transpose()
df.to_csv('Classification_Report.csv', index = True)
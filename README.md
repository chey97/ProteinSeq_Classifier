
Classifying a protein's family based on the protein sequence.

1) Import Dataset

Before running this model you must dowload the relevant datset
[DATA_SET](https://drive.google.com/drive/folders/1K_3DtAUWUvlC-b20Sg3SBnvdnSlZRee0?usp=share_link)

This is a protein data set retrieved from Research Collaboratory for Structural Bioinformatics (RCSB) Protein Data Bank (PDB).

The PDB archive is a repository of atomic coordinates and other information describing proteins and other important biological macromolecules. Structural biologists use methods such as X-ray crystallography, NMR spectroscopy, and cryo-electron microscopy to determine the location of each atom relative to each other in the molecule. They then deposit this information, which is then annotated and publicly released into the archive by the wwPDB.

The constantly-growing PDB is a reflection of the research that is happening in laboratories across the world. This can make it both exciting and challenging to use the database in research and education. Structures are available for many of the proteins and nucleic acids involved in the central processes of life, so you can go to the PDB archive to find structures for ribosomes, oncogenes, drug targets, and even whole viruses. However, it can be a challenge to find the information that you need, since the PDB archives so many different structures. You will often find multiple structures for a given molecule, or partial structures, or structures that have been modified or inactivated from their native form.

There are two data files. Both are arranged on "structureId" of the protein:

pdb_data_no_dups.csv contains protein meta data which includes details on protein classification, extraction methods, etc.

data_seq.csv contains >400,000 protein structure sequences.

Original Dataset can be dowloaded from [RCSB-PDB](https://www.rcsb.org)

2) Filter and Process Data

I found out that its better use CountVectorizer -- a feature extractor that is usually used with NLP machine learning models.
The data is loaded into two seperate pandas dataframes and filtering ( filtering the datasets where the classification is equal to 'Protein', followed by removing all other variables other than structureId and sequence for the data_seq_csv, and structureId and classification in the no_dups dataset. ), projection, and a join is performed to get the data together.

Since the data-set appears to be a wide distribution of counts for family types filter it out for having a certain amount of recordes that are of a specific family type.Filter out for 1,000 family types that will allow a machine learning model to learn a pattern for a specific class.

3) Train Test Split

After filtering the dataset, a split on the data to create a training and testing set must be performed. 
After splitting the data, utilize the CountVectorizer to create a dictionary composed from the training dataset and it will extract individual characters or subsets of characters to gain features.


4) Machine Learning Models

Multinomial Naive Bayes approach works well for these types of count vectorized features. Adaboost was also used but the it appears that Naive Bayes model does better in classification than Adaboost model.

5) Visualize Metrics

A visualization of a confusion matrix and a clasification report for the Navie Bayes prediction shows that the label index 3 being misclassified as index 38 quite a bit. Based on the names listed below, it makes sense for these two to be confused.

![Heat_Map](https://github.com/chey97/ProteinSeq_Classifier/blob/47242111421619c21bc7565cdbb50fc1cbfa70ad/plots/Heat%20Map.png)

The relavent classification report : ![Classification_Report](https://github.com/chey97/ProteinSeq_Classifier/blob/31e409b8496bd1927c7730cdc84312c0a08cb7a2/Classification_Report.csv)

Reasons for Model Error : 

Proteins in general can be a type of enzyme, or a signaling protein, structural, and various other choices. A lof of proteins tend to share very similar characteristics, as some proteins are meant to bind in similar regions as others. For example, a Hydrolase enzyme and a Hydrolase inhibitor protein are going to have similar structures as they will target very similar areas. This is reflected in the confusion matrix and heat map. Gene regulator proteins will have a similarity to RNA binding proteins, DNA binding proteins, as well as transcription proteins. The biggest thing to note as well, as the model only uses features of 4 amino acids at most. The possibility of utilizing amino acids of higher degree in theory should be able to create an even higher accuracy.

Future Work : 

There is definitely room for improvement for the model. Utilizing factors such as pH, molecular weight, and other components may be able to yield more information on family group. Furthermore, if possible, increase the length of the ngram_range to include more than just 4 characters to allow for higher interaction between the amino acids as reflected in reality

This modal was build for academic purpose - DNASeq - Protein Sequence Classifier Copyright (C) 2023  Chethiya Galkaduwa


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

The relavent classification report : 

|FIELD1                                 |precision          |recall            |f1-score          |support           |
|---------------------------------------|-------------------|------------------|------------------|------------------|
|HYDROLASE                              |0.518957345971564  |0.7906137184115524|0.626609442060086 |277.0             |
|TRANSFERASE                            |0.6575342465753424 |0.8571428571428571|0.7441860465116279|224.0             |
|OXIDOREDUCTASE                         |0.7                |0.7957166392092258|0.7447956823438704|607.0             |
|IMMUNE SYSTEM                          |0.6691729323308271 |0.6926070038910506|0.6806883365200764|514.0             |
|LYASE                                  |0.8985115020297699 |0.7830188679245284|0.8367989918084435|848.0             |
|HYDROLASE/HYDROLASE INHIBITOR          |0.6461038961038961 |0.868995633187773 |0.7411545623836127|229.0             |
|TRANSCRIPTION                          |0.6381578947368421 |0.8151260504201681|0.7158671586715868|357.0             |
|VIRAL PROTEIN                          |0.7253731343283583 |0.7653543307086614|0.7448275862068966|635.0             |
|TRANSPORT PROTEIN                      |0.5855513307984791 |0.7427652733118971|0.6548547129695252|622.0             |
|VIRUS                                  |0.9061032863849765 |0.9747474747474747|0.9391727493917273|198.0             |
|SIGNALING PROTEIN                      |0.6988636363636364 |0.8065573770491803|0.7488584474885844|305.0             |
|ISOMERASE                              |0.5135135135135135 |0.9712460063897763|0.6718232044198895|313.0             |
|LIGASE                                 |0.7880280148829065 |0.7703252032520326|0.7790760575570702|9348.0            |
|MEMBRANE PROTEIN                       |0.6617368221463784 |0.7790178571428571|0.7156038548287881|2240.0            |
|PROTEIN BINDING                        |0.8758265980896399 |0.7534766118836915|0.8100577641862046|3164.0            |
|STRUCTURAL PROTEIN                     |0.9352580927384077 |0.8712306438467807|0.9021097046413503|1227.0            |
|CHAPERONE                              |0.8778625954198473 |0.8017928286852589|0.838105153565851 |1004.0            |
|STRUCTURAL GENOMICS, UNKNOWN FUNCTION  |0.9391381608174145 |0.8775425487754255|0.9072961373390559|2409.0            |
|SUGAR BINDING PROTEIN                  |0.715922107674685  |0.6243756243756243|0.6670224119530417|1001.0            |
|DNA BINDING PROTEIN                    |0.7068403908794788 |0.7045454545454546|0.7056910569105692|616.0             |
|PHOTOSYNTHESIS                         |0.6545454545454545 |0.7659574468085106|0.7058823529411765|235.0             |
|ELECTRON TRANSPORT                     |0.5264900662251656 |0.6186770428015564|0.5688729874776386|257.0             |
|TRANSFERASE/TRANSFERASE INHIBITOR      |0.9338726697150896 |0.7858517093384638|0.8534919231696536|6757.0            |
|METAL BINDING PROTEIN                  |0.3359683794466403 |0.8823529411764706|0.4866412213740458|289.0             |
|CELL ADHESION                          |0.7609001406469761 |0.9031719532554258|0.8259541984732826|599.0             |
|UNKNOWN FUNCTION                       |0.5549242424242424 |0.6116910229645094|0.5819265143992055|958.0             |
|PROTEIN TRANSPORT                      |0.7785467128027682 |0.9336099585062241|0.8490566037735849|241.0             |
|TOXIN                                  |0.6704730831973899 |0.7611111111111111|0.7129228100607112|540.0             |
|CELL CYCLE                             |0.5981651376146789 |0.8402061855670103|0.6988210075026795|388.0             |
|RNA BINDING PROTEIN                    |0.7527985074626866 |0.6289945440374123|0.6853503184713375|1283.0            |
|DE NOVO PROTEIN                        |0.48513986013986016|0.7939914163090128|0.6022788931090614|699.0             |
|HORMONE                                |0.8269841269841269 |0.5947488584474886|0.6918990703851262|876.0             |
|GENE REGULATION                        |0.8441926345609065 |0.8882265275707899|0.8656499636891795|671.0             |
|OXIDOREDUCTASE/OXIDOREDUCTASE INHIBITOR|0.7779816513761468 |0.8514056224899599|0.8130393096836052|498.0             |
|APOPTOSIS                              |0.8232258064516129 |0.7160493827160493|0.765906362545018 |1782.0            |
|MOTOR PROTEIN                          |0.7815620298180823 |0.7838134430727023|0.7826861173892198|7290.0            |
|PROTEIN FIBRIL                         |0.3458646616541353 |0.7796610169491526|0.4791666666666667|590.0             |
|METAL TRANSPORT                        |0.772093023255814  |0.8258706467661692|0.7980769230769231|201.0             |
|VIRAL PROTEIN/IMMUNE SYSTEM            |0.8602878916172735 |0.6065671641791045|0.711484593837535 |1675.0            |
|CONTRACTILE PROTEIN                    |0.720292504570384  |0.7490494296577946|0.7343895619757689|526.0             |
|FLUORESCENT PROTEIN                    |0.8908396946564886 |0.6634451392836839|0.7605083088954057|1759.0            |
|TRANSLATION                            |0.4212218649517685 |0.6121495327102804|0.4990476190476191|214.0             |
|BIOSYNTHETIC PROTEIN                   |0.8675149700598802 |0.886085626911315 |0.8767019667170952|1308.0            |
|accuracy                               |0.7682970559759027 |0.7682970559759027|0.7682970559759027|0.7682970559759027|
|macro avg                              |0.7126125724642683 |0.7797415285472434|0.7331245199167302|55774.0           |
|weighted avg                           |0.7929027805102244 |0.7682970559759027|0.7741092058773669|55774.0           |
|--------------------------------------------------------------------------------------------------------------------|

6) Reasons for Model Error : 

Proteins in general can be a type of enzyme, or a signaling protein, structural, and various other choices. A lof of proteins tend to share very similar characteristics, as some proteins are meant to bind in similar regions as others. For example, a Hydrolase enzyme and a Hydrolase inhibitor protein are going to have similar structures as they will target very similar areas. This is reflected in the confusion matrix and heat map. Gene regulator proteins will have a similarity to RNA binding proteins, DNA binding proteins, as well as transcription proteins. The biggest thing to note as well, as the model only uses features of 4 amino acids at most. The possibility of utilizing amino acids of higher degree in theory should be able to create an even higher accuracy.

7) Future Work : 

There is definitely room for improvement for the model. Utilizing factors such as pH, molecular weight, and other components may be able to yield more information on family group. Furthermore, if possible, increase the length of the ngram_range to include more than just 4 characters to allow for higher interaction between the amino acids as reflected in reality

This modal was build for academic purpose - DNASeq - Protein Sequence Classifier Copyright (C) 2023  Chethiya Galkaduwa

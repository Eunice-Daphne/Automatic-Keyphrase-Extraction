CS582 INFORMATION RETRIEVAL 
RESEARCH PROJECT 

TEAM MEMBERS : 

Name : Anjana Anand 
netid : aanand31
UIN : 672420136

Name : Eunice Daphne John Kanagaraj
netid : ejohnk2
UIN : 670581874

*************Instruction to run the code**************

Go to the directory where the project is stored in Command Prompt. Then enter the following commands to get the results.

-> Key2Vec with ARXIV Model 

	In command prompt : 
  	>> python Key2Vec(ARXIV).py 

-> Key2Vec with WIKI Model : 

	In command prompt : 
  	>> python Key2Vec(WIKI).py

-> PageRank 

	In command prompt : 
  	>> python PageRank.py


**********OUTPUT*********** 
-> Key2Vec with ARXIV Model 

Total no. of documents evaluated: 425
Recall at Top 5 is 0.06321
Precision at Top 5 is 0.06945
F1-score at Top 5 is 0.06618

Recall at Top 10 is 0.08304
Precision at Top 10 is 0.06457
F1-score at Top 10 is 0.07265

Recall at Top 15 is 0.10021
Precision at Top 15 is 0.06265
F1-score at Top 15 is 0.0771


-> Key2Vec with WIKI Model
	
Total no. of documnets evaluated: 425
Recall at Top 5 is 0.07236
Precision at Top 5 is 0.07808
F1-score at Top 5 is 0.07511

Recall at Top 10 is 0.09205
Precision at Top 10 is 0.06708
F1-score at Top 10 is 0.07761

Recall at Top 15 is 0.1074
Precision at Top 15 is 0.06276
F1-score at Top 15 is 0.07922 

-> PageRank 

Recall at Top 5 is 0.06779
Precision at Top 5 is 0.06447
F1-score at Top 5 is 0.06609

Recall at Top 10 is 0.09678
Precision at Top 10 is 0.04613
F1-score at Top 10 is 0.06248

Recall at Top 15 is 0.10393
Precision at Top 15 is 0.0351
F1-score at Top 15 is 0.05248

*************Data Set**************

The data set is given in the "DATASET-WWW" folder which contains the following directories and files. 
The abstracts are given both with and without POS TAGS in the "contentsubset-pos" and "contentsubset" folders respectively.
The gold standards are given in the "gold" folder. 
The list of files containing the file name for evaluation are given the "queries.overlap.list" file.
The stopword list is given the stopwords.text 

*************The Embedding Models**************

The two embedding models are given in the "Embedding models" folder which contains the following directories.
The "Fasttext_ARXIV" contains the pre-trained fasttext model on ARXIV data. 
The "Fasttext_WIKI" contains the pre-trained fasttext model downloaded from the fasttext website. 

  



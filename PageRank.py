#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from nltk.stem import PorterStemmer
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import stopwords
from heapq import nlargest
import math
import os
import sys

gold_list = []

#List of files needed for evaluation
list_file = open(os.path.join(os.getcwd(),'DATASET-WWW','queries.overlap.list'))

filename_overlap = []
for file in list_file:
    filename_overlap.append(file.strip('\n'))
    
#path to abstract files in the dataset
dataset_abstract_path = os.path.join(os.getcwd(),'DATASET-WWW','contentsubset-pos')      #os.path.join(dataset_path,'www','abstracts')

#path to gold standard files in the dataset
dataset_gold_path = os.path.join(os.getcwd(),'DATASET-WWW','gold')  #os.path.join(dataset_path,'www','gold')

#path to stopwords list
stopWord_list_path = os.path.join(os.getcwd(),'DATASET-WWW','stopwords.txt')
with open(stopWord_list_path) as stopWordsFile:
        stopWords = [line.rstrip('\n') for line in stopWordsFile]
        
#get window size as input
window_size = 1        

#List containing names of files that has gold standards in the dataset
for filename in os.listdir(dataset_gold_path):
    gold_list.append(filename)
    
def Abstract_tokenizer(pathToFiles, window):
    pathToFiles = pathToFiles
    vocabulary = {}
    numberOfWords = 0
    wordCount = {}
    #stopWords = stopwords.words('english')
    stemmer = PorterStemmer()
    #list of nouns and adjectives to keep
    Nouns_and_Adj_list = ['NN','NNS','NNP','NNPS','JJ']
    window=window
    ngrams = {}
    for filename in os.listdir(pathToFiles):
        #check if the file has gold standard if not continue to next file
        if filename not in filename_overlap:
            continue
        with open(os.path.join(pathToFiles, filename)) as currentFile:
            ngrams[filename] = Undirected_graph()
            wordCount[filename] = {}
            tokens_in_window = []
            for line in currentFile:
                #Tokenize on whitespace
                tokens = WhitespaceTokenizer().tokenize(line)
                for token in tokens:
                    token = token.split("_")
                    #only consider words that are in the list given and are not stopwords
                    if token[1] not in Nouns_and_Adj_list or token[0].lower() in stopWords:
                        tokens_in_window = []
                        continue
                    token = token[0].lower()
                    # Stemming using Porter Stemer
                    token = stemmer.stem(token)
                    if token not in wordCount[filename]:
                        if token not in vocabulary:
                            vocabulary[token] = 1
                        else:
                            vocabulary[token] += 1
                        ngrams[filename].add_node(token)
                        wordCount[filename][token] = 1
                    else:
                        wordCount[filename][token] += 1
                    #construct graph based on the window size and words that are adjacent within the window
                    if tokens_in_window:
                        for lastToken in tokens_in_window:
                            currentEdge = ngrams[filename].get_edge(lastToken, token)
                            if currentEdge == -1:
                                ngrams[filename].add_edge(lastToken, token, 1)
                            else:
                                ngrams[filename].add_edge(lastToken, token, currentEdge+1)
                    tokens_in_window.append(token)
                    if len(tokens_in_window) > window:
                        tokens_in_window = tokens_in_window[1:]
    return ngrams
            
class Undirected_graph:
    def __init__(self):
        self.graph = {}        
    #add a node to the graph
    def add_node(self,node):
        if node not in self.graph:
            self.graph[node] = {}
    #add an edge to the graph
    def add_edge(self, first_node, second_node, weight):
        if first_node not in self.graph:
            self.add_node(first_node)
        if second_node not in self.graph:
            self.add_node(second_node)
        self.graph[first_node][second_node] = weight
        self.graph[second_node][first_node] = weight
    #returns the edge between the given nodes
    def get_edge(self, first_node, second_node):
        if first_node in self.graph:
            if second_node in self.graph[first_node]:
                return self.graph[first_node][second_node]
        return 0

def N_gram_Tokenizer(pathToFiles,n):
    pathToFiles = pathToFiles
    stemmer = PorterStemmer()
    n=n                             #n in n-gram
    ngrams = {}
    #check if the file has gold standard if not continue to next file
    for filename in os.listdir(pathToFiles):
        if filename not in filename_overlap:
            continue
        with open(os.path.join(pathToFiles, filename)) as currentFile:
            ngrams[filename] = {}
            tokens_in_window = []
            for line in currentFile:
                #Tokenize on whitespace
                tokens = WhitespaceTokenizer().tokenize(line)
                for token in tokens:
                    token = token.split("_")
                    token = token[0].lower()  
                    token = stemmer.stem(token)
                    tokens_in_window.append(token)
                    if len(tokens_in_window) > n:
                        tokens_in_window = tokens_in_window[1:]
                    newNGram = ''
                    if len(tokens_in_window) == n:
                        for currentToken in tokens_in_window:
                            newNGram = '{} {}'.format(newNGram,currentToken)
                        newNGram = newNGram.strip()
                    if newNGram:
                        ngrams[filename][newNGram] = 0
    return ngrams                      

def Page_Rank(graph,alpha,convergence):
    p = {}
    page_rank = {}
    last_page_rank = {}
    for node in graph.graph:
        p[node] = 1/len(graph.graph)
        page_rank[node] = 1/len(graph.graph)
        last_page_rank[node] = 1/len(graph.graph)
    for it in range(0,convergence):
        for vi in graph.graph:
            page_rank[vi] = alpha*sum(graph.get_edge(vj,vi)/sum(graph.get_edge(vj,vk) for vk in graph.graph[vj])*last_page_rank[vj] for vj in graph.graph[vi])+(1-alpha)*p[vi]
        #normalization and updating steps
        total_weight = sum(page_rank[vi]**2 for vi in page_rank)**(1/2)
        for vi in page_rank:
            page_rank[vi] = page_rank[vi] / total_weight
            last_page_rank[vi] = page_rank[vi]
    return page_rank

def MRR(pathToFiles):
        pathToFiles = pathToFiles
        stemmer = PorterStemmer()
        ngrams = {}
        for filename in os.listdir(pathToFiles):
            with open(os.path.join(pathToFiles, filename), encoding="utf8", errors="ignore") as currentFile:
                lines = [line.strip('\n') for line in currentFile]
                ngrams[filename] = {}
                tokens_in_window = []
                for line in lines:
                    #tokenize on whitespace
                    tokens = WhitespaceTokenizer().tokenize(line)
                    tokens_in_window = []
                    for token in tokens:
                        token = token.split("_")
                        token = token[0].lower()
                        #stemming using Porter Stemmer
                        token = stemmer.stem(token)
                        if not token:
                            continue
                        tokens_in_window.append(token)
                    newNGram = ''
                    for currentToken in tokens_in_window:
                        newNGram = '{} {}'.format(newNGram,currentToken)
                    newNGram = newNGram.strip()
                    if newNGram:
                        ngrams[filename][newNGram] = 0
        return ngrams

#Calculating Recall
def recall1(gold_list, retkey):
        if type(retkey) is list and type(gold_list) is list:
            ret_rel_docs = [x for x in gold_list for y in retkey if x==y]
            num_ret_rel_docs = len(ret_rel_docs)
            num_rel_docs = len(gold_list)
            r = num_ret_rel_docs/num_rel_docs
            return r
        
#Calculating Precision        
def precision1(gold_list, retkey):
        if type(retkey) is list and type(gold_list) is list:
            ret_rel_docs=[x for x in gold_list for y in retkey if x==y]
            no_ret_rel=len(ret_rel_docs)
            no_ret=len(retkey)
            p=no_ret_rel/no_ret
            return p
        
#Creates a graph of words
abstract_token = Abstract_tokenizer(dataset_abstract_path, window=window_size)

#PageRank score for each word
doc_score = {}
convergence = 10
for graph in abstract_token:
    doc_score[graph] = Page_Rank(abstract_token[graph], 0.85, convergence)

#Retrieving 1-grams, 2-grams, 3-grams from the text.
doc_MultiNGrams = {}
for i in range(1,2):
    doc_MultiNGrams[i] =  N_gram_Tokenizer(dataset_abstract_path,n=i)

#Joining the ngrams into one dictionary with their summed score.
doc_NGrams = {}
for n in doc_MultiNGrams:
    for f in doc_MultiNGrams[n]:
        ngrams = doc_MultiNGrams[n][f]
        for ngram in ngrams:
            words = ngram.split(' ')
            ngram_score = 0
            for word in words:
                if word in doc_score[f]:
                    ngram_score += doc_score[f][word]
            doc_MultiNGrams[n][f][ngram] = ngram_score
        if f not in doc_NGrams:
            doc_NGrams[f] = {}
        doc_NGrams[f] = {**doc_NGrams[f],**doc_MultiNGrams[n][f]}

#Retrieves the top k phrases with highest scores and calculates the MRR
golden_standards = MRR(dataset_gold_path)
MRR = {}
for k in range(5,20,5):
    p=[]
    r=[]
    for file in doc_NGrams:
        top = nlargest(k, doc_NGrams[file], key=doc_NGrams[file].get)
        g_list = []
        for key, value in golden_standards[file].items() :
            g_list.append(key)
        recall = recall1(g_list,top)
        precision = precision1(g_list,top)
        r.append(recall)
        p.append(precision)
    avg_p=round(sum(precision for precision in p)/len(p),5)
    avg_r=round(sum(recall for recall in r)/len(r),5)
    f1_score= round((2*avg_p*avg_r / (avg_p+avg_r)),5)
    print("Recall at Top",k,"is",avg_r)
    print("Precision at Top",k,"is",avg_p)
    print("F1-score at Top",k,"is",f1_score)
    print("\n")


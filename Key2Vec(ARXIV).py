#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Pre-processing Candidate keyphrases
def split_into_sentences(input_text):
    for i, sentence in enumerate(nlp(input_text).sents):
        #sentence Split    
        doc = nlp(sentence.text)
        for token in doc:
            token1 = re.sub("[^\P{P}-]+","",str(token))
            text = token1.lower()   
            text = re.sub("[^\W+|\W+$]","",str(text))
            if text not in stopwords and text:
                unigram.append([token1, token.idx])
            else:
                continue
                
        #Named Entity Extraction
        for entity in nlp(sentence.text).ents: 
            #removing date,time,percent,money,quantity,ordinal,cardinal
            if entity.label_ not in entity_filter:
                token1 = re.sub("[^\P{P}-]+","",str(entity))
                text = token1.lower().split(" ") 
                try:
                    if text[0] in stopwords or nltk.pos_tag(text[0]) in tags or nltk.pos_tag(text[0]) in dt:
                        text.remove(text[0])
                    if text[-1] in stopwords or nltk.pos_tag(text[-1]) in tags:
                        text.remove(text[-1])
                #Very few phrases become null during this step hence the error is handled here
                except IndexError:
                    #print(entity.text)
                    #print(filename)
                    indexerror = 0
                text1 = ' '.join(text)
                #checking if the entire entity phrase is digit or not
                flag = all(token.isdigit() for token in text1)
                if not flag:
                    entitylist.append([text1, entity.start_char, entity.end_char])     
        
        #Noun Phrase Extraction
        for np in nlp(sentence.text).noun_chunks:
            #converting to str and lowercase and removing the first and last stopwords 
            token1 = re.sub("[^\P{P}-]+","",str(np))
            text = token1.lower().split(" ") 
            if text[0] in stopwords or nltk.pos_tag(text[0]) in tags or nltk.pos_tag(text[0]) in dt:
                text.remove(text[0])
            text1 = ' '.join(text)
            #checking if the entire noun phrase is digit or not
            flag = all(token.isdigit() for token in text1)
            if not flag:
                noun_phraselist.append([text1, np.start_char, np.end_char])             
                
        #Remove dupicates from Noun Phrase and Named Entity
        for i in entitylist:
            for j in noun_phraselist:
                if i[1]==j[1] and i[2]==j[2]:
                    noun_phraselist.remove(j)
        entity_and_nounphrase = entitylist + noun_phraselist
        
        #Remove unigrams based on offset from Noun Phrase and Named Entity
        for i in unigram:
            for j in entity_and_nounphrase:
                start=j[1]
                end=j[2]
                if i[1]>=start and i[1]<=end:
                    i[0]=""
        for i in unigram:
            if not i[0]=="":
                entity_and_nounphrase.append(i)
                
    #Removing leading and trailing whitespaces
    for i in entity_and_nounphrase:
        entity_and_nounphrase_list.append(i[0].lstrip().rstrip())


# In[2]:


#Pre-processing for Theme Phrases
def theme_phrases(input_text):
    for i, sentence in enumerate(nlp(input_text).sents):
        if i>1:
            break
        doc = nlp(sentence.text)
        for token in doc:
            token1 = re.sub("[^\P{P}-]+","",str(token))
            #if not token.is_stop and token.is_alpha:
            text = token1.lower() 
            text = re.sub("[^\W+|\W+$]","",str(text))
            if text not in stopwords and text:
                unigram1.append([token1, token.idx])
            else:
                continue
                
        #print(unigram1)
        for entity in nlp(sentence.text).ents: 
            #removing date,time,percent,money,quantity,ordinal,cardinal
            if entity.label_ not in entity_filter:
                token1 = re.sub("[^\P{P}-]+","",str(entity))
                text = token1.lower().split(" ") 
                try:
                    if text[0] in stopwords or nltk.pos_tag(text[0]) in tags or nltk.pos_tag(text[0]) in dt:
                        text.remove(text[0])
                    if text[-1] in stopwords or nltk.pos_tag(text[-1]) in tags:
                        text.remove(text[-1])
                #Very few phrases become null during this step hence the error is handled here
                except IndexError:
                    #print(entity.text)
                    #print(filename)
                    indexerror = 0
                text1 = ' '.join(text)
                #checking if the entire entity phrase is digit or not
                flag = all(token.isdigit() for token in text1)
                if not flag:
                    entitylist1.append([text1, entity.start_char, entity.end_char])     
        
        #Noun Phrase Extraction
        for np in nlp(sentence.text).noun_chunks:
            #converting to str and lowercase and removing the first and last stopwords 
            token1 = re.sub("[^\P{P}-]+","",str(np))
            text = token1.lower().split(" ") 
            if text[0] in stopwords or nltk.pos_tag(text[0]) in tags or nltk.pos_tag(text[0]) in dt:
                text.remove(text[0])
            text1 = ' '.join(text)
            #checking if the entire noun phrase is digit or not
            flag = all(token.isdigit() for token in text1)
            if not flag:
                noun_phraselist1.append([text1, np.start_char, np.end_char])               
                
        #Remove dupicates from Noun Phrase and Named Entity
        for i in entitylist1:
            for j in noun_phraselist1:
                if i[1]==j[1] and i[2]==j[2]:
                    noun_phraselist1.remove(j)
        entity_and_nounphrase1 = entitylist1 + noun_phraselist1
        
        #Remove unigrams based on offset from Noun Phrase and Named Entity
        for i in unigram1:
            for j in entity_and_nounphrase1:
                start=j[1]
                end=j[2]
                if i[1]>=start and i[1]<=end:
                    i[0]=""
        for i in unigram1:
            if not i[0]=="":
                entity_and_nounphrase1.append(i)
    
    #Removal of leading and trailing whitespaces 
    for i in entity_and_nounphrase1:
        entity_and_nounphrase_list1.append(i[0].lstrip().rstrip())


# In[3]:


#Point-wise Mutual Information Calculation
def pmi(x, y):
    x_count = _Fdist[x]
    y_count = _Fdist[y]
    prob_x = x_count / len(_Fdist)
    prob_y = y_count / len(_Fdist)
    cc = co_occur(x,y)
    prob_xy = cc / len(_Fdist)
    if prob_xy==0:
        pmi = 0
    else:
        pmi = np.log(prob_xy / (prob_x*prob_y))
    return pmi

#Co-occurence or semantic similarity calculation
def co_occur(x,y):
    count = 0
    for i in range(0,len(entity_and_nounphrase_list)):
        if i==0:
            if entity_and_nounphrase_list[i]==x and entity_and_nounphrase_list[i+1]==y:
                count=count+1
        elif i==len(entity_and_nounphrase_list)-1:
            if entity_and_nounphrase_list[i]==x and entity_and_nounphrase_list[i-1]==y:
                count=count+1              
        elif entity_and_nounphrase_list[i]==x and entity_and_nounphrase_list[i-1]==y or entity_and_nounphrase_list[i+1]==y:
            count=count+1           
    return count

#Calculating the edge weights in the graph
def score1(x,y):
    x_vec = [model.wv.get_vector(x)]
    y_vec = [model.wv.get_vector(y)]
    cos_sim = cosine_similarity(y_vec, x_vec).ravel()
    if cos_sim[0]>=1:
        cos_sim[0]=0.99
    semantic = 1 / (1 - cos_sim[0])
    cooccur = pmi(x,y)
    sr = semantic * cooccur
    return sr


# In[4]:


#Bi-directional Graph
class directed_graph:
    def __init__(self):
        self.G = nx.DiGraph()

    #Add node to the graph
    def add_node_1(self,node):
        if not self.G.has_node(node):
            self.G.add_node(node)

    #Add edge to the graph
    def add_edge_1(self,first_node, second_node, weight):
        if not self.G.has_node(first_node):
            self.G.add_node(first_node)
        if not self.G.has_node(second_node):
            self.G.add_node(second_node)
        self.G.add_edge(first_node, second_node, weight=weight)
        self.G.add_edge(second_node, first_node, weight=weight)

    #Get the edge of the graph
    def get_edge_1(self,first_node, second_node):
        if self.G.has_node(first_node):
            if second_node in self.G[first_node]:
                return self.G.adj[first_node][second_node]['weight']
        return 0
    
    #Draw the graph using networkx draw()
    def draw_graph(self):
        nx.draw(self.G, with_labels = True)
        plt.draw() 


# In[5]:


#Calculating the theme-weighted PageRank score
def s(vj, graph, damp, p, convergence, last_page_rank):
    second = 0
    first = (1-damp)*score[vj][0]
    for vk in graph.G[vj]:
        count = graph.G.out_degree(vk)
        second  = second + ( float( score1(vk,vj)/ count) * last_page_rank[vk])
    rank =  first + second 
    return rank
def page_rank(graph,damp,convergence,p=None):
    p = {}
    page_rank = {}
    last_page_rank = {}
    for node in graph.G:
        p[node] = 1/len(graph.G)
        page_rank[node] = 1/len(graph.G)
        last_page_rank[node] = 1/len(graph.G)
    for iteration in range(0,convergence):
        for vi in graph.G:
            page_rank[vi] = s(vi, graph, damp, p, convergence, last_page_rank)
        #normalization and updating steps
        total_weight = sum(page_rank[vi]**2 for vi in page_rank)**(1/2)
        for vi in page_rank:
            page_rank[vi] = page_rank[vi] / total_weight
            last_page_rank[vi] = page_rank[vi]
    return page_rank


# In[6]:


#Calculate Recall
def recall1(gold_list, retkey):
        if type(retkey) is list and type(gold_list) is list:
            ret_rel_docs = [x for x in gold_list for y in retkey if x==y]
            num_ret_rel_docs = len(ret_rel_docs)
            num_rel_docs = len(gold_list)
            r = num_ret_rel_docs/num_rel_docs
            return r

#Calculate Precision
def precision1(gold_list, retkey):
        if type(retkey) is list and type(gold_list) is list:
            ret_rel_docs=[x for x in gold_list for y in retkey if x==y]
            no_ret_rel=len(ret_rel_docs)
            no_ret=len(retkey)
            p=no_ret_rel/no_ret
            return p 


# In[7]:


import glob
import spacy
import en_core_web_sm
import nltk
import regex as re
import os
import sys
import operator
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import stopwords
from gensim.models import FastText
from gensim.test.utils import common_texts
from sklearn.metrics.pairwise import cosine_similarity
from operator import itemgetter
import itertools
import numpy as np
from itertools import islice
import networkx as nx
import matplotlib.pyplot as plt

nlp = en_core_web_sm.load()

#Stopwords list
stopwords = set(stopwords.words('english'))
#Types of Named Entities to be removed
entity_filter = ['DATE', 'TIME','PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL','CARDINAL']

#Punctuations to be removed
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~''' 

#Types of Noun Phrases to be removed
tags = ['INTJ','CCONJ','ADP','DET','NUM','PART','PRON','SCONJ','PUNCT','SYM','X']
dt = ['DT']

#Loading the fasttext ARXIV model
model = FastText.load(os.path.join(os.getcwd(),'Embedding models','Fasttext_ARXIV','fasttext_arxiv_small.model'))

#Path to the gold standard for www
dataset_gold_path = os.path.join(os.getcwd(),'DATASET-WWW','gold')

#Store the filenames of gold standard
gold_list_name = []
for filename in os.listdir(dataset_gold_path):
    gold_list_name.append(filename)

#Path to the dataset    
pathToFiles = os.path.join(os.getcwd(),'DATASET-WWW','contentsubset')

count=0

ngrams = {}
wordCount = {}
document_scores = {}
sorted_keyphrase = {}
convergence = 10
top_n_documents=[5,10,15]
p={}
r={}

#List of files needed for evaluation
list_file = open(os.path.join(os.getcwd(),'DATASET-WWW','queries.overlap.list'))

#Store the filenames of files for evaluation
filename_overlap = []
for file in list_file:
    filename_overlap.append(file.strip('\n'))

gold_list = {}

#Store the gold standards in a dict with filename and corresponding keyphrases
for filename in os.listdir(dataset_gold_path):
        with open(os.path.join(dataset_gold_path, filename)) as currentFile:
            if filename in filename_overlap:
                gold_list[filename] = []
                for keyphrase in currentFile:
                    kp = keyphrase.strip('\n')
                    gold_list[filename].append(kp)

#Main part of code to call other functions for Keyphrase Extraction                    
for filename in os.listdir(pathToFiles):
        if filename not in gold_list:
            continue
        count=count+1
        #print(count," ",filename)
        #Candidate Phrases Extraction
        unigram = []
        entitylist = []
        noun_phrase = []
        entity_and_nounphrase = []
        entire_bow = []
        noun_phraselist = []
        entity_and_nounphrase_list = []
        #Theme Phrases Extraction
        unigram1 = []
        entitylist1 = []
        noun_phrase1 = []
        entity_and_nounphrase1 = []
        entire_bow1 = []
        noun_phraselist1 = []
        entity_and_nounphrase_list1 = []
        with open(os.path.join(pathToFiles, filename)) as currentFile:  
            textfile = ""
            for line in currentFile:
                tokens = WhitespaceTokenizer().tokenize(line)
                for token in tokens:
                    token1 = token.split("_")
                    textfile= textfile+" "+token1[0]
        #Text pre-processing for Candidate Keyphrases
        split_into_sentences(textfile)
        for i in entity_and_nounphrase_list[:]:   #added to remove single punctuations
            if len(i)<=1:
                entity_and_nounphrase_list.remove(i)
        #Text pre-processing for Thematic Keyphrases
        theme_phrases(textfile)
        #Thematic vector representatio
        themed_vector = 0
        for i in entity_and_nounphrase_list1[:]:
            try:
                themed_vector += model.wv.get_vector(i)
            #Very few phrases don't have a vector in the model hence the error is handled here
            except KeyError:
                #print(i)
                entity_and_nounphrase_list1.remove(i)
        #Candidate Keyphrase Vector Representation
        for i in entity_and_nounphrase_list[:]:
            try:
                raw = model.wv.get_vector(i)
            #Very few phrases don't have a vector in the model hence the error is handled here
            except KeyError:
                #print(i)
                entity_and_nounphrase_list.remove(i)
        score = {}
        for i in entity_and_nounphrase_list[:]:
            x = [model.wv.get_vector(i)]
            y = [themed_vector]
            #Calulate cosine similarity between theme vector and candidate Keyphrases
            cos = cosine_similarity(y, x).ravel()
            score[i]=cos
        _Fdist  =  nltk . FreqDist (entity_and_nounphrase_list)
        ngrams[filename] = directed_graph()
        wordCount[filename] = {}
        lastTokens = []  
        #Graph Construction
        for token in entity_and_nounphrase_list:
            if token not in wordCount[filename]:
                ngrams[filename].add_node_1(token)
                wordCount[filename][token] = 1
            else:
                wordCount[filename][token] += 1
            if lastTokens:
                for lastToken in lastTokens:
                    currentEdge = ngrams[filename].get_edge_1(lastToken, token)
                    if currentEdge == -1:
                        ngrams[filename].add_edge_1(lastToken, token, 1)
                    else:
                        ngrams[filename].add_edge_1(lastToken, token, currentEdge+1)
            lastTokens.append(token)
            if len(lastTokens) > 4:
                lastTokens = lastTokens[1:] 
        document_scores[filename] = page_rank(ngrams[filename], 0.85, convergence)
        #Ranked list of keyphrases for each of the documnets
        sorted_keyphrase[filename] = sorted(document_scores[filename].items(), key=operator.itemgetter(1), reverse=True)
        #break
print("Total no. of documents evaluated:",count)

for i in top_n_documents:
    p=[]
    r=[] 
    for filename, keyphrases in sorted_keyphrase.items():
        retdoc=keyphrases[:i]
        keyphrases = []
        for k, v in retdoc:
            keyphrases.append(k)
        #Calculating the recall
        recall = recall1(gold_list[filename],keyphrases)
        r.append((filename,recall))
        #Calculating the precision
        precision = precision1(gold_list[filename],keyphrases)
        p.append((filename,precision))
    #Calculating the average recall
    avg_p=round(sum(precision for filename,precision in p)/len(p),5)
    #Calculating the average precision
    avg_r=round(sum(recall for filename,recall in r)/len(r),5)
    #Calculating F1-score
    f1_score= round((2*avg_p*avg_r / (avg_p+avg_r)),5)
    print("Recall at Top",i,"is",avg_r)
    print("Precision at Top",i,"is",avg_p)   
    print("F1-score at Top",i,"is",f1_score)
    print("\n")    


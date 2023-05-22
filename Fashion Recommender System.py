#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import re
import urllib

from tqdm import tqdm
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from bs4 import BeautifulSoup
from skimage import io
import os
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity  
from sklearn.metrics import pairwise_distances
warnings.filterwarnings("ignore")


# In[2]:


data = pd.read_csv('Fashion Dataset.csv' , on_bad_lines='skip')


# In[3]:


data.drop(['Unnamed: 0', 'p_id', 'p_attributes'] , axis = 1 , inplace = True)


# In[4]:


data.head()


# In[5]:


data.dropna(how='all' , inplace = True)


# In[6]:


data[data['colour'].isnull()]


# In[7]:


data.drop([367,2458,14129] , axis = 0 , inplace = True)


# In[8]:


stop_words = {'a',
 'about',
 'above',
 'after',
 'again',
 'against',
 'ain',
 'all',
 'am',
 'an',
 'and',
 'any',
 'are',
 'aren',
 "aren't",
 'as',
 'at',
 'be',
 'because',
 'been',
 'before',
 'being',
 'below',
 'between',
 'both',
 'but',
 'by',
 'can',
 'couldn',
 "couldn't",
 'd',
 'did',
 'didn',
 "didn't",
 'do',
 'does',
 'doesn',
 "doesn't",
 'doing',
 'don',
 "don't",
 'down',
 'during',
 'each',
 'few',
 'for',
 'from',
 'further',
 'had',
 'hadn',
 "hadn't",
 'has',
 'hasn',
 "hasn't",
 'have',
 'haven',
 "haven't",
 'having',
 'he',
 'her',
 'here',
 'hers',
 'herself',
 'him',
 'himself',
 'his',
 'how',
 'i',
 'if',
 'in',
 'into',
 'is',
 'isn',
 "isn't",
 'it',
 "it's",
 'its',
 'itself',
 'just',
 'll',
 'm',
 'ma',
 'me',
 'mightn',
 "mightn't",
 'more',
 'most',
 'mustn',
 "mustn't",
 'my',
 'myself',
 'needn',
 "needn't",
 'no',
 'nor',
 'not',
 'now',
 'o',
 'of',
 'off',
 'on',
 'once',
 'only',
 'or',
 'other',
 'our',
 'ours',
 'ourselves',
 'out',
 'over',
 'own',
 're',
 's',
 'same',
 'shan',
 "shan't",
 'she',
 "she's",
 'should',
 "should've",
 'shouldn',
 "shouldn't",
 'so',
 'some',
 'such',
 't',
 'than',
 'that',
 "that'll",
 'the',
 'their',
 'theirs',
 'them',
 'themselves',
 'then',
 'there',
 'these',
 'they',
 'this',
 'those',
 'through',
 'to',
 'too',
 'under',
 'until',
 'up',
 've',
 'very',
 'was',
 'wasn',
 "wasn't",
 'we',
 'were',
 'weren',
 "weren't",
 'what',
 'when',
 'where',
 'which',
 'while',
 'who',
 'whom',
 'why',
 'will',
 'with',
 'won',
 "won't",
 'wouldn',
 "wouldn't",
 'y',
 'you',
 "you'd",
 "you'll",
 "you're",
 "you've",
 'your',
 'yours',
 'yourself',
 'yourselves'}


# In[9]:


def preprocess(text):
    
    text =  re.sub(r'[\W]', ' ', text)  # remove special chracter
    text = re.sub(' +' , ' ' , text)
    text = text.lower()
    text = re.sub(r'\d', '', text)

    text = text.split()

    text = [word for word in text if word not in stop_words]
    text = [word for word in text if len(word)>1]
    text = ' '.join(text)
    text = text.lstrip()
    text = text.rstrip()

    return text


# In[10]:


def preprocess_description(text):
    
    text = re.sub(r'<[^>]+>','',text)
    text = re.sub(r'[\W]', ' ', text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r' +', ' ', text)
    text = text.lower()
    
    text = text.split()
    text = [word for word in text if word not in stop_words]
    text = [word for word in text if len(word)>1]
    text = ' '.join(text)
    text = text.lstrip()
    text = text.rstrip()
    
    
    return text


# In[11]:


data['name'] = data['name'].apply(lambda x : preprocess(x))
data['description'] = data['description'].apply(lambda x : preprocess_description(x) )

data['brand'] = data['brand'].apply(lambda x : x.lower())
data['brand'] = data['brand'].apply(lambda x : re.sub(' ' , '_' , x))

data['colour'] = data['colour'].apply(lambda x : x.lower())
data['colour'] = data['colour'].apply(lambda x : re.sub(' ','_',x))


# In[12]:


data['colour'].value_counts()


# In[13]:


data.head()


# # Extracting Features

# #### TF-IDF Model on Title Feature

# In[14]:


tfidf_title_vectorizer = TfidfVectorizer()
tfidf_title_features   = tfidf_title_vectorizer.fit_transform(data['name'])
print(tfidf_title_features.get_shape()) # number of rows and columns in feature matrix.

# for tf-idf w2v model
title_idf = dict(zip(tfidf_title_vectorizer.get_feature_names_out() , list(tfidf_title_vectorizer.idf_)))
title_tfidf_words = set(tfidf_title_vectorizer.get_feature_names_out())


# #### TF-IDF W2V Model On Title Feature

# In[24]:


from numpy import asarray
# load the whole embedding into memory
embeddings_index = dict()
f = open('glove.6B.100d.txt',encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))


# In[26]:


tfidf_title_w2v_vectors = [] # tf-idf-w2v for each sen will stored in this list

for sentence in tqdm(data['name']):
    
    tf_idf_sum = 0
    sent_vec = np.zeros(100) # we want 300d vector representation for each sentence
    
    for each_word in sentence.split():
        
        if each_word in embeddings_index.keys() and each_word in title_tfidf_words:
            # getting 100d vector
            vec = embeddings_index[each_word]
            # vocabulary[each_word] = idf of that word in corpus
            # sentence.count(word) gives tf value in essay
            tf_idf = title_idf[each_word] * (sentence.count(each_word)/len(sentence))
            sent_vec += (vec * tf_idf)
            tf_idf_sum += tf_idf
    
    if tf_idf_sum != 0:
        sent_vec = sent_vec/tf_idf_sum
    
    tfidf_title_w2v_vectors.append(sent_vec)

tfidf_title_w2v_vectors =  np.asarray(tfidf_title_w2v_vectors)


# #### TF-IDF Model on Description Feature

# In[29]:


tfidf_descrption_vectorizer = TfidfVectorizer()
tfidf_descrption_features   = tfidf_descrption_vectorizer.fit_transform(data['description'])
print(tfidf_descrption_features.get_shape()) # number of rows and columns in feature matrix.

# for tf-idf w2v model
description_idf = dict(zip(tfidf_descrption_vectorizer.get_feature_names_out() , list(tfidf_descrption_vectorizer.idf_)))
description_tfidf_words = set(tfidf_descrption_vectorizer.get_feature_names_out())


# #### TF-IDF-W2V Model on Description Feature

# In[30]:


tfidf_description_w2v_vectors = [] # tf-idf-w2v for each sen will stored in this list

for sentence in tqdm(data['description']):
    
    tf_idf_sum = 0
    sent_vec = np.zeros(100) # we want 300d vector representation for each sentence
    
    for each_word in sentence.split():
        
        if each_word in embeddings_index.keys() and each_word in description_tfidf_words:
            # getting 100d vector
            vec = embeddings_index[each_word]
            # vocabulary[each_word] = idf of that word in corpus
            # sentence.count(word) gives tf value in essay
            tf_idf = description_idf[each_word] * (sentence.count(each_word)/len(sentence))
            sent_vec += (vec * tf_idf)
            tf_idf_sum += tf_idf
    
    if tf_idf_sum != 0:
        sent_vec = sent_vec/tf_idf_sum
    
    tfidf_description_w2v_vectors.append(sent_vec)

tfidf_description_w2v_vectors =  np.asarray(tfidf_description_w2v_vectors)


# #### Binary BOW model Colour Feature

# In[16]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(binary = True)
color_binary_feature = vectorizer.fit_transform(data['colour'])


# #### Binary BOW model Brand Feature

# In[17]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(binary = True)
brand_binary_feature = vectorizer.fit_transform(data['brand'])


# ### Create Recommendation Model

# In[18]:


def create_single_feature_recommendation_model(id , feature_vector ):
    
    pairwise_dist = pairwise_distances(feature_vector,feature_vector[id].reshape(1,-1))
    indices = np.argsort(pairwise_dist.flatten())[0:17]
    pdists  = np.sort(pairwise_dist.flatten())[0:17]
    #print(pdists)

    df_indices = list(data.index[indices])[1:]
    df_image = data.loc[df_indices]

    image_data = []
    image_name = []
    for ele in df_image['img']:
        image_data.append(ele)

    for ele in df_image['name']:
        image_name.append(ele)


    plt.rcParams["figure.figsize"] = [5,5]
    plt.rcParams["figure.autolayout"] = True

    f = data['img'][id]
    a = io.imread(f)

    plt.imshow(a)
    plt.axis('off')
    plt.title("*****************************************************************       Name : {}       *****************************************************************".format(data['name'][id]))
    plt.show()

    print()
    print()
    print()

  
    print("======================================================= Recommendations =======================================================")

    print()
    print()
    print()

    f, axarr = plt.subplots(5,3, figsize=(30, 30))
    count = 0

    for i in range(5):
        for j in range(3):

            urllib.request.urlretrieve(image_data[count], 'a.png')
            img = Image.open("a.png")
            img.thumbnail((400, 400))
            axarr[i,j].set_title(image_name[count])
            axarr[i,j].imshow(img)
            count +=1 



# In[19]:


def create_recommendation_from_same_brand(id ,title_feature_vector , brand_feature_vector ,tfidf_descrption_features ):
  
    title_pairwise_dist = pairwise_distances(title_feature_vector,title_feature_vector[id].reshape(1,-1))
    brand_pairwise_dist = pairwise_distances(brand_feature_vector,brand_feature_vector[id].reshape(1,-1))
    #color_pairwise_dist = pairwise_distances(color_feature_vector,color_feature_vector[id].reshape(1,-1))
    descrption_pairwise_dist = pairwise_distances(tfidf_descrption_features,tfidf_descrption_features[id].reshape(1,-1))


    sum_of_pairwise_distance = 0.2 * title_pairwise_dist + 0.2 * descrption_pairwise_dist + 0.6 * brand_pairwise_dist

    indices = np.argsort(sum_of_pairwise_distance.flatten())[0:17]
    pdists  = np.sort(sum_of_pairwise_distance.flatten())[0:17]
    #print(pdists)

    df_indices = list(data.index[indices])[1:]
    df_image = data.loc[df_indices]

    image_data = []
    image_name = []
    for ele in df_image['img']:
        image_data.append(ele)

    for ele in df_image['name']:
        image_name.append(ele)


    plt.rcParams["figure.figsize"] = [5,5]
    plt.rcParams["figure.autolayout"] = True

    f = data['img'][id]
    a = io.imread(f)

    plt.imshow(a)
    plt.axis('off')
    plt.title("*****************************************************************       Name : {}       *****************************************************************".format(data['name'][id]))
    plt.show()

    print()
    print()
    print()

  
    print("======================================================= Recommendations =======================================================")

    print()
    print()
    print()

    f, axarr = plt.subplots(5,3, figsize=(30, 30))
    count = 0

    for i in range(5):
        for j in range(3):

            urllib.request.urlretrieve(image_data[count], 'a.png')
            img = Image.open("a.png")
            img.thumbnail((400, 400))
            axarr[i,j].set_title(image_name[count])
            axarr[i,j].imshow(img)
            count +=1 



# In[39]:


def create_recommendation_model(id , title_feature_vector , brand_feature_vector, color_feature_vector , tfidf_descrption_features ):
  
    title_pairwise_dist = pairwise_distances(title_feature_vector,title_feature_vector[id].reshape(1,-1))
    brand_pairwise_dist = pairwise_distances(brand_feature_vector,brand_feature_vector[id].reshape(1,-1))
    color_pairwise_dist = pairwise_distances(color_feature_vector,color_feature_vector[id].reshape(1,-1))
    descrption_pairwise_dist = pairwise_distances(tfidf_descrption_features,tfidf_descrption_features[id].reshape(1,-1))


    sum_of_pairwise_distance = 0.4 * title_pairwise_dist + 0.5 * descrption_pairwise_dist + 0.1 * color_pairwise_dist

    indices = np.argsort(sum_of_pairwise_distance.flatten())[0:17]
    pdists  = np.sort(sum_of_pairwise_distance.flatten())[0:17]
    #print(pdists)

    df_indices = list(data.index[indices])[1:]
    df_image = data.loc[df_indices]

    image_data = []
    image_name = []
    for ele in df_image['img']:
        image_data.append(ele)

    for ele in df_image['name']:
        image_name.append(ele)


    plt.rcParams["figure.figsize"] = [5,5]
    plt.rcParams["figure.autolayout"] = True

    f = data['img'][id]
    a = io.imread(f)

    plt.imshow(a)
    plt.axis('off')
    plt.title("*****************************************************************       Name : {}       *****************************************************************".format(data['name'][id]))
    plt.show()

    print()

  
    print("======================================================= Recommendations =======================================================")
    print()

    f, axarr = plt.subplots(5,3, figsize=(30, 30))
    count = 0

    for i in range(5):
        for j in range(3):

            urllib.request.urlretrieve(image_data[count], 'a.png')
            img = Image.open("a.png")
            img.thumbnail((400, 400))
            axarr[i,j].set_title(image_name[count])
            axarr[i,j].imshow(img)
            count +=1 



# In[38]:


create_recommendation_model(10 , tfidf_title_w2v_vectors , brand_binary_feature , color_binary_feature , tfidf_description_w2v_vectors)


# In[40]:


create_recommendation_model(10 , tfidf_title_features , brand_binary_feature , color_binary_feature , tfidf_descrption_features)


# In[ ]:





# In[35]:


create_recommendation_from_same_brand(10, tfidf_title_features , brand_binary_feature ,tfidf_descrption_features )


# In[37]:


create_recommendation_from_same_brand(10, tfidf_title_w2v_vectors , brand_binary_feature ,tfidf_description_w2v_vectors )


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





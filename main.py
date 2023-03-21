import re
import sys
import ast
import stanza
import spacy
import operator
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# for ids extraction
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import nltk
nltk.download('punkt')

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


from features_extraction import get_features
from utils import dublicated_k, get_ngrams, extract_ids, get_cluster_number, get_cat_model

dataset_path = sys.argv[1]
category_name = sys.argv[2]
test_lang = sys.argv[3]

tweets = pd.read_csv(dataset_path)

tweet_ids = tweets['id']


text_features, sim_features, laser_features = get_features(tweets['text'], category_name, test_lang)

TRANSFORMER_DIM = 1024 #laser

# - - - - - FEATURES  reshaping- - - - -
laser_features_class = tf.reshape(laser_features, [-1, 1, TRANSFORMER_DIM])
text_features_class = tf.reshape(text_features, [-1, 1, 15])


print('Data shapes:', laser_features_class.shape, text_features_class.shape)

# loading model and classification

model = tf.keras.models.load_model('resources/disaster_detect')
model.summary()

true_index = []

predict = model.predict([laser_features_class, text_features_class])

for num in range(len(predict)):
    if np.argmax(predict[num]) == 1:
        true_index.append(num)

true_x = tweets['text'].reset_index(drop=True)
true = true_x.loc[true_index]

true_laser_features = laser_features[true_index]
true_sim_features = [sim_features[num] for num in true_index]
true_text_features = [text_features[num] for num in true_index]

print('test size:', len(true_laser_features), print(true_sim_features), print(true_text_features))


cat_model = tf.keras.models.load_model('resources/category_model/' + get_cat_model(category_name))
cat_model.summary()

laser_features_class = tf.reshape(true_laser_features, [-1, 1, TRANSFORMER_DIM])
text_features_class = tf.reshape(true_text_features, [-1, 1, 15])
sim_features_class = tf.reshape(true_sim_features, [-1, 1, 6])




# - - - - - SUMMARIZATION - - - - -

param_K = 200
# create dictionary for storing number of row and probability
higher_prob = {}
for num, prob in enumerate(cat_model.predict([text_features_class, laser_features_class, sim_features_class])):
    higher_prob[num] = prob[0][1]

# sort dictionary by probability decreasing
sorted_output = sorted(higher_prob.items(),key=operator.itemgetter(1),reverse=True) # sorting dictionary and choose the top of probable answers

# get list with top K entities
ents = [ent[0] for ent in sorted_output[:param_K]]

# create string (text_4sum) with the most relevant tweets
texts_for_sum = []

ids_for_sum = [] # store ids of tweets for using them for summary evaluation interface

for ent in ents:
    #print(final_texts[ent])
    ids_for_sum.append(tweet_ids[ent])
    text = re.sub('#', '', tweets['text'][ent])
    text = re.sub(r'((www\.[\s]+)|(https?://[^\s]+))', '', text)
    text = re.sub('@ ?[A-Za-z0-9]+', '', text)
    text = re.sub('^[A-Za-z0-9]+', '', text)
    text = re.sub('^ ?RT', '', text)
    text = re.sub('^ ?: ', '', text)
    texts_for_sum.append(text)



vectorizer = TfidfVectorizer(min_df=2)
new_texts = vectorizer.fit_transform(texts_for_sum)

new_texts = new_texts.toarray() # transform data for use with sklearn

#define number of clusters
number_of_clusters = get_cluster_number(new_texts)

# define the model
clust_model = KMeans(n_clusters=number_of_clusters, random_state=42)
# fit the model
clust_model.fit(new_texts)


# create ordering list based on frequency
nums = {}
for n in clust_model.labels_:
    if n not in nums:
        nums[n] = 1
    else:
        nums[n] += 1
nums = dict(sorted(nums.items(), key=lambda item: item[1], reverse=True))
ordered_list = list(nums.keys())




# - - - - - - - - - - - - - - - - - - - - 

sum_model = AutoModelForSeq2SeqLM.from_pretrained("t5-large")
tokenizer = AutoTokenizer.from_pretrained("t5-large")

# Ordered
summary = ''

for num in ordered_list:
    cluster_text = ''

    cluster_ids = [] # store ids for extracting original tweets
    cluster_tweets = [] # store texts of tweets for extracting original tweets
    
    for n, cluster in enumerate(clust_model.labels_):
        if cluster == num:
            cluster_text += texts_for_sum[n]
            cluster_ids.append(ids_for_sum[n])
            cluster_tweets.append(texts_for_sum[n])
    
    if len(summary) < 600:

        inputs = tokenizer("summarize: " + cluster_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = sum_model.generate(inputs["input_ids"], max_length=100, min_length=40, length_penalty=4.0, num_beams=4, early_stopping=True)
        generated_summary = tokenizer.decode(outputs[0])[6:-4]

        extract_ids(generated_summary, cluster_ids, cluster_tweets, 3)

        summary += generated_summary + '\n' + extract_ids(generated_summary, cluster_ids, cluster_tweets, 3)

print(summary)

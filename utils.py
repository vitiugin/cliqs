from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import operator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def dublicated_k(entities, embeddings):
    dublicates = []
    for num1 in range(len(entities)):
        for num2 in range(len(entities)):
            if entities[num1] != entities[num2]:
                if cosine_similarity(embeddings[entities[num1]], embeddings[entities[num2]]) > 0.9:
                    dublicates.append(num2)
    
    dublicates_wo_dublicates = list(set(dublicates))

    # drop dublicates from entities
    entities_wo_dublicates = [item for item in entities if item not in dublicates_wo_dublicates]
    
    return(entities_wo_dublicates[:100])


# - - - - - TEST IDs EXTRACTOR - - - - -


def get_ngrams(text, n):
    n_grams = ngrams(word_tokenize(text), n)
    return [ ' '.join(grams) for grams in n_grams]


def extract_ids(f_summary, ids, tweets, n):

    #print(len(ids), len(tweets))
    summary_ngrams = get_ngrams(f_summary, n)

    sim_counter = {}
    for num in range(len(tweets)):
        count = 0
        for ngram in summary_ngrams:
            if ngram in get_ngrams(tweets[num], n):
                count += 1
        
        if count > 0:
            sim_counter[ids[num]] = count # add ids of tweet in l
    
    sorted_sims = dict(sorted(sim_counter.items(), key=operator.itemgetter(1),reverse=True))

    # http://twitter.com/anyuser/status/
    top_ids = list(sorted_sims.keys())[:3] # top-3 similar tweets
    
    ids_string = ''
    for id in top_ids:
        ids_string += 'http://twitter.com/anyuser/status/'+ str(id) + '\n'
    
    return ids_string



def get_cluster_number(text_dataset):
    coefficients = []
    for num in range(2, 6):
        # define the model
        clust_model = KMeans(n_clusters=num)
        # fit the model
        clust_model.fit(text_dataset)
        # assign a cluster to each example
        yhat = clust_model.predict(text_dataset)
        #print(silhouette_score(text_dataset, yhat))
        coefficients.append(silhouette_score(text_dataset, yhat))
    
    max_value = max(coefficients)
    max_index = coefficients.index(max_value)
    return(max_index+2)


def get_cat_model(cat_name):

    cats = {'Casualties': 'sum_casualties', 'Damage': 'sum_damage', 
            'Danger': 'sum_danger', 'Sensor': 'sum_sensor', 
            'Service': 'sum_service', 'Weather': 'sum_weather'}

    if cat_name in cats.keys():
        return cats[cat_name]
    else:
        print('ERROR. requested category doesn\'t support.')
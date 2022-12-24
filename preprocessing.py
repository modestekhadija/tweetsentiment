import numpy as np
import pandas as pd
import gensim
import spacy
import re
from spacy.tokens import Token
from gensim.models import KeyedVectors
from joblib import load
nlp = spacy.load('en_core_web_sm')


# Load w2v KeyVectors
lemm_w2v_vectors = KeyedVectors.load("embedding/lemm_word2vec.wordvectors", mmap='r')

# Loading the models
log_regression = load('model/best_regression_model_saved.joblib')

def clean(text):
    nlp.Defaults.stop_words -= {'not', 'never'}
    
    # user_name regular expression
    regex_user = r"@[\w\d_]+"
    is_user = lambda token: re.fullmatch(regex_user, token.text)
    Token.set_extension("is_user", getter=is_user, force=True)

    doc = nlp(text)
    tokens = [token.lemma_ .lower()
              for token in doc 
              if (not token.is_punct)     # supprimer les ponctuations
              and (len(token)>1)          # supprimer les token de longueur 1
              and (not token.is_stop)     # supprimer les stopwords
              and (not token.is_space)    # supprimer les espaces vides
              and (not token.is_digit)    # supprimer les nombres
              and (not token.like_email)  # supprimer les emails
              and (not token.like_url)    # supprimer les URLs
              and (not token._.is_user)]  # supprimer les noms d'utilisateurs

    return ' '.join(tokens)


# Vectoriser un token avec word2vec
def word_vector(token, model_vectors, size=200):

    vec = np.zeros(size).reshape((1, size))
    count = 0
    
    for word in token:
        try:
            vec += model_vectors[word].reshape((1, size))
            count += 1.
            
        except KeyError: 
            continue
    if count != 0:
        vec /= count
    return vec

# Text vectorisation
def vectorisation(tweet, model_vectors=lemm_w2v_vectors, size=200):
    print('Tokenisation ...')
    sentence = gensim.utils.simple_preprocess(tweet)

    print('Vectorisation du corpus ...')
  
    vector = word_vector(model_vectors = model_vectors, 
                                   token = sentence, 
                                   size = size)
        
    print('matrix created')

    return vector

def predict_text(tweet):
    cleaned_tweet = clean(tweet)
    input_data = vectorisation(cleaned_tweet)
    prediction = log_regression.predict(input_data)
    
    return 'NEGATIVE' if prediction==0 else 'POSITIVE'
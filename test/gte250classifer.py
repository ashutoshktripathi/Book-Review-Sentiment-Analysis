import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import nltk

embedding = "../saved_models/use5"
hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           dtype=tf.string, trainable=False)


# load the saved tf model
tfmodel = tf.keras.models.load_model('../tfmodel')


def tokenize_sentence(text_list):
    c = 0
    res = []
    for i in text_list:
        sen = nltk.sent_tokenize(i)
        res.append(sen)
        c += 1
        if c % 10000 == 0:
            print(f"{c} items tokenized")
    return res

def pad_for_lstm(series,maxlen = 50):
    return tf.keras.preprocessing.sequence.pad_sequences(series,dtype="float64", maxlen = maxlen) 

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
def prepare_data_for_prediction(text_list, summ_list, verbose = False):
    train = tokenize_sentence(text_list)
    
    # custome and fast way to get the embeddings
    count = 0
    train_emb = []
    for i in chunks(train, 10):
        flat_chunk = []
        for j in i:
            flat_chunk += j

        em = hub_layer(flat_chunk).numpy()
        c = 0
        f = []
        for j in i:
            f.append(em[c: c + len(j)])
            c+=len(j)

        train_emb += f
        count += 1
        if verbose:
            print(f"{count * 10} texts embedded")

    train_emb_pad = pad_for_lstm(train_emb, 50)
    
    
    # for summary
    count = 0
    train_summ_emb = []
    for i in chunks(list(summ_list), 100):
        em = hub_layer(i).numpy()
        train_summ_emb.append(em)
        count+=1
        if verbose:
            if count % 100 == 0:
                print(f"{count * 100} sumary embedded")
    train_X2_emb = np.concatenate(train_summ_emb)
    
    return train_emb_pad, train_X2_emb
    
def classify(text_list, summ_list, verbose = False):
    reviewEmb , summaryEmb = prepare_data_for_prediction(text_list, summ_list,verbose)
    pred = tfmodel.predict([reviewEmb , summaryEmb]).argmax(axis = 1)
    sentiment = []
    for i in pred:
        if i == 0:
            sentiment.append('negative')
        elif i == 1:
            sentiment.append('neutral')
        else:
            sentiment.append('positive')
    return sentiment
'''
This is the training script for sentiment analyis for large data with word count greater than 250.
This is designed to work on combination of two feaures. 1st, the actual comment or body which is expected to be very large. 2nd, an additional summary or title which is expected to be small.'''

import warnings 
warnings.filterwarnings("ignore")
# import the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_hub as hub
import numpy as np
import os
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder
import nltk


def load_universal_sentence_encoder(path):
    '''
    loads the TF-FUB USE pre-trained model.
    param:
        path: either provide path of downloaded folder or link to the tf-hub repo.
    '''
    hub_layer = hub.KerasLayer(path, input_shape=[], 
                           dtype=tf.string, trainable=False)
    return hub_layer


def tokenize_sentence(text_list):
    '''
    function to tokenize sentence
    param:
        test_list : list or numpy array containing text
    '''
    count = 0
    res = []
    for i in text_list:
        sen = nltk.sent_tokenize(i)
        res.append(sen)
        count += 1
        if count % 10000 == 0:
            print(f"{c} items tokenized")
    return res

def pad_for_lstm(series,maxlen = 50):
    '''
    pad the sequence using tf.keras.preprocessing.sequence.pad_sequences
    '''
    return tf.keras.preprocessing.sequence.pad_sequences(series,dtype="float64", maxlen = maxlen) 

def chunks(lst, n):
    """Yield successive n-sized chunks from list."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
def prepare_data_for_prediction(review_list, summ_list, hub_layer, verbose = True, pad_max_lenght = 50):
    review = tokenize_sentence(review_list)
    # custome and fast way to get the embeddings
    count = 0
    review_emb = []
    for i in chunks(review, 10):
        flat_chunk = []
        for j in i:
            flat_chunk += j

        em = hub_layer(flat_chunk).numpy()
        c = 0
        f = []
        for j in i:
            f.append(em[c: c + len(j)])
            c+=len(j)

        review_emb += f
        count += 1
        if count % 100 == 0:
            if verbose:
                print(f"*****{count * 10} texts embedded*****")

    review_emb_pad = pad_for_lstm(review_emb, pad_max_lenght)
    
    
    # for summary
    count = 0
    review_summ_emb = []
    for i in chunks(list(summ_list), 100):
        em = hub_layer(i).numpy()
        review_summ_emb.append(em)
        count+=1
        if verbose:
            if count % 100 == 0:
                print(f"*****{count * 100} sumary embedded*****")
    review_summ_emb = np.concatenate(review_summ_emb)
    
    return review_emb_pad, review_summ_emb


class Attention(tf.keras.layers.Layer):
    def __init__(self, step_dim=120,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = tf.keras.initializers.get('glorot_uniform')

        self.W_regularizer = tf.keras.regularizers.get(W_regularizer)
        self.b_regularizer = tf.keras.regularizers.get(b_regularizer)

        self.W_constraint = tf.keras.constraints.get(W_constraint)
        self.b_constraint = tf.keras.constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape = (input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape = (input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


def run(
    train_review, 
    train_summary, 
    test_review, 
    test_summary, 
    train_label, 
    test_label , 
    num_labels,
    use_path,
    save_model_path,
    use_dim = 512,
    pad = 50,
    test_split = .2, 
    lstm_depth = 500, 
    lstm_dropout = .25, 
    recurrent_dropout = .25, 
    dropout = .3,
    input1_dense_layers = (256,128,64),
    input2_dense_layers = (256,128,64),
    dense_activation = "relu",
    output_activation = "softmax",
    learning_rate = 2e-5,
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    save_monitor='val_accuracy',
    save_mode='max',
    earlystopping_monitor='val_accuracy',
    earlystopping_patience=5,
    epochs = 25,
    batch_size=128,
):
    
    # load universal sentnece encoder
    hub_layer = load_universal_sentence_encoder(use_path)
    print("**hub use model loaded**")
    
    # get the train and test review and summary embedding
    print("**preparing train data**")
    train_review_emb, train_summ_emb = prepare_data_for_prediction(train_review, train_summary, hub_layer)
    print("**train data preperation done**")
    print("**preparing test data**")
    test_review_emb, test_summ_emb = prepare_data_for_prediction(test_review, test_summary, hub_layer)
    print("**test data preperation done**")
    # do one hot encoding on the train label
    y = OneHotEncoder(sparse=False).fit_transform(train_label.reshape(-1, 1))
    
    # perform the train test split
    X_train, X_val, X_train2, X_val2, y_train, y_val = train_test_split(train_review_emb, train_summ_emb, y, test_size=test_split, random_state=42)
    
    # create the model
    
    # create input 1 model
    
    input1 = tf.keras.layers.Input(shape=(pad, use_dim))
    x = tf.keras.layers.Masking(mask_value=0.)(input1)
    x = tf.keras.layers.LSTM(lstm_depth,dropout=lstm_dropout, recurrent_dropout=recurrent_dropout,return_sequences=True)(input1)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = (Attention(pad))(x)
    
    for i in input1_dense_layers[:-1]:
        x = tf.keras.layers.Dense(i, activation=dense_activation)(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(input1_dense_layers[-1], activation=dense_activation)(x)
    
    
    # create input 2 model
    input2 = tf.keras.Input(shape=(use_dim,))
    for i in input2_dense_layers[:-1]:
        add = tf.keras.layers.Dense(i, activation=dense_activation)(input2)
        add = tf.keras.layers.Dropout(dropout)(add)
    add = tf.keras.layers.Dense(input2_dense_layers[-1], activation=dense_activation)(add)
    
    
    # concat input1 and input2
    merged = tf.keras.layers.Concatenate()([x,add])
    preds = tf.keras.layers.Dense(num_labels, activation=output_activation)(merged)
    
    # finish the model
    model = tf.keras.models.Model(inputs=[input1, input2], outputs=preds)
    
    # print model summary
    
    print(model.summary())
    
    # compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),loss=loss,metrics=metrics)
    
    checkpoint = save_model_path
    try:
        os.mkdir(checkpoint)
    except:
        pass
    checkpoint_filepath  = f'{checkpoint}/'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor=save_monitor,
        mode=save_mode,
        save_best_only=True)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor=earlystopping_monitor, patience=earlystopping_patience)
    
    history = model.fit(
    [X_train, X_train2], y_train,
    epochs=epochs,
    validation_data=([X_val,X_val2], y_val),
    batch_size=batch_size,
    verbose=1,
    shuffle=True,
    callbacks=[model_checkpoint_callback, early_stopping]
)
    
    # test the accuracy on test set
    
    # laod the best save model
    saved_model = tf.keras.models.load_model(checkpoint)
    
    # predict on the test set
    y_pred = saved_model.predict([test_review_emb, test_summ_emb])
    
    #print the accuracy
    print(accuracy_score(y_pred.argmax(axis = 1) , test_label))
    
    # generate the classification report
    cr = classification_report( test_label ,y_pred.argmax(axis = 1) ,output_dict=True)
    print(cr)
    
    # save the classfication report in the checkpoint folder
    pd.DataFrame(cr).to_csv(f"{checkpoint}/classification_report.csv") 
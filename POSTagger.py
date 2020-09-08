import tensorflow as tf 
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation,Dropout,GRU,BatchNormalization
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix,classification_report
 
class POSTagger:
    # Custom Accuracy Metric - Ignoring predictions for "-PAD-" tag.
    def ignore_pad_accuracy(self,to_ignore=0):
        def custom_accuracy(y_true, y_pred):
            y_true_class = K.argmax(y_true, axis=-1)
            y_pred_class = K.argmax(y_pred, axis=-1)
    
            ignore_mask = K.cast(K.not_equal(y_pred_class, to_ignore), 'int32')
            matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
            accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
            return accuracy
        return custom_accuracy

    def __init__(self,embedding_dim,wordset,tagset,embedding_matrix,max_length):
        super(POSTagger,self).__init__()

        self.embedding_dim = embedding_dim
        #self.hidden_size = hidden_size
        self.wordset = wordset
        self.tagset = tagset
        self.MAX_LENGTH = max_length
        self.embedding_matrix = embedding_matrix
        # Define network Architecture
        print("Defining Model Architecture")
        self.model = Sequential()
        self.model.add(InputLayer(input_shape=(self.MAX_LENGTH, )))
        #self.model.add(Embedding(len(self.wordset)+1),self.embedding_dim,input_length = self.MAX_LENGTH)
        self.model.add(Embedding(len(self.wordset)+1, self.embedding_dim,input_length=self.MAX_LENGTH ,weights=[self.embedding_matrix],trainable=False))
        self.model.add(Dropout(0.10))
        self.model.add(Bidirectional(GRU(256, return_sequences=True)))
        #self.model.add(Dropout(0.15))
        self.model.add(TimeDistributed(Dense(len(self.tagset))))
        self.model.add(Activation('softmax'))

    def compile_model(self):
        self.model.compile(loss='categorical_crossentropy',optimizer=Adam(0.001),metrics=['accuracy',self.ignore_pad_accuracy(0)])
        print(self.model.summary())
    # Label Encoding
    def to_categorical(self,sequences, categories):
        self.cat_sequences = []
        for s in sequences:
            cats = []
            for item in s:
                cats.append(np.zeros(categories))
                cats[-1][item] = 1.0
            self.cat_sequences.append(cats)   
        return np.array(self.cat_sequences)

    def train_model(self,train_sentences_X,train_tags_y,dev_sentences_X,dev_tags_y,num_epochs=40):
        self.compile_model()
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_custom_accuracy', patience=5)
        self.model.fit(train_sentences_X,
            self.to_categorical(train_tags_y, len(self.tagset)),
            batch_size=128,
            epochs=num_epochs,
            validation_data=(dev_sentences_X,self.to_categorical(dev_tags_y,len(self.tagset))),
            callbacks = [early_stop]
         )
        return self.model
    def predict(self,eval_sentences_X):
        return self.model.predict(eval_sentences_X)

    def evaluate(self,eval_sentences_X,eval_tags_y,train_vocab_y):
        scores = self.model.evaluate(eval_sentences_X,self.to_categorical(eval_tags_y,len(train_vocab_y)))
        print("Evaluation Accuracy: "+f"{model.metrics_names[2]}: {scores[2] * 100}")
        return scores
    
    def logits_to_tokens(self,sequences, index):
        token_sequences = []
        for categorical_sequence in sequences:
            print(categorical_sequence)
            token_sequence = []
            for categorical in categorical_sequence:
                token_sequence.append(index[np.argmax(categorical)])
            token_sequences.append(token_sequence)
        return token_sequences 

    def show_metrics(self,eval_words,eval_tags_y):
        y_true = eval_tags_y
        predictions = self.predict(eval_words)
        y_pred = np.apply_along_axis(np.argmax,2,predictions)
        print("Classification_Report on Dev data set:\n",classification_report(y_true.flatten(),y_pred.flatten()))
    
    def save_model(self):
        self.model.save("models/POS_Tagger",save_format='tf')
        print("Model saved at: models/POS_Tagger") 






    
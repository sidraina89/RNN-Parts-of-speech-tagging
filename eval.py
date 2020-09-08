import train
import dataload
import joblib
import tensorflow as tf
import POSTagger
import sys
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K

def convert_token_to_number(df_list,vocab,word_or_tag="WORD"):
    token_ids = []
    pos = int(word_or_tag=="TAG")
    for df in df_list:
        s_int = []
        for tok in df[pos]:
            try:
                if pos == 0:
                    tok = tok.lower()
                s_int.append(vocab[tok])
            except:
                if pos == 0:
                    s_int.append(vocab["-OOV-"])
        token_ids.append(s_int)
    return token_ids
def ignore_pad_accuracy(to_ignore=0):
        def custom_accuracy(y_true, y_pred):
            y_true_class = K.argmax(y_true, axis=-1)
            y_pred_class = K.argmax(y_pred, axis=-1)
    
            ignore_mask = K.cast(K.not_equal(y_pred_class, to_ignore), 'int32')
            matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
            accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
            return accuracy
        return custom_accuracy

def to_categorical(sequences, categories):
        cat_sequences = []
        for s in sequences:
            cats = []
            for item in s:
                cats.append(np.zeros(categories))
                cats[-1][item] = 1.0
            cat_sequences.append(cats)   
        return np.array(cat_sequences)
def evaluate(model,test_sentences_X,test_tags_y,train_vocab_y):
        scores = model.evaluate(test_sentences_X,to_categorical(test_tags_y,len(train_vocab_y)))
        print("Test Accuracy: "+f"{model.metrics_names[2]}: {scores[2] * 100}")
        return scores

def build_test_data(test_file=None):
    raw_data_list = dataload.load_raw_conllu(None,None,test_file)
    test_data = dataload.convert2df(raw_data_list)[0]
    # Build vocabulary from words and tags in training set
    test_df_list = [[d["form"].tolist(), d["upos"].tolist()] for d in test_data]
    test_sentences_X = convert_token_to_number(test_df_list,train_vocab_X,"WORD")
    test_tags_y = convert_token_to_number(test_df_list,train_vocab_y,"TAG")
    return test_sentences_X,test_tags_y

def pad_sequence_list(seq_list,MAX_LENGTH):
    return [pad_sequences(seq, maxlen=MAX_LENGTH, padding='post') for seq in seq_list]

def load_model():
        model = tf.keras.models.load_model('models/POS_Tagger',custom_objects={'custom_accuracy':ignore_pad_accuracy(0)},)
        return model

if __name__ == "__main__":
    if len(sys.argv) ==2:
        #print(sys.argv)
        test_file = sys.argv[1]
        train_vocab_X = joblib.load("vocab/word_to_idx")
        train_vocab_y = joblib.load("vocab/tag_to_idx")
        MAX_LENGTH = joblib.load("vocab/Train_max_len")

        test_sentences_X, test_tags_y = build_test_data(test_file)
        test_sentences_X, test_tags_y = pad_sequence_list([test_sentences_X, test_tags_y],MAX_LENGTH)
        pos = load_model()
        evaluate(pos,test_sentences_X,test_tags_y,train_vocab_y)
        #preds = np.apply_along_axis(np.argmax,2,pos.predict(test_sentences_X))
        #print(preds[0])
        #print(test_tags_y[0])
        #sentence_acc = []
        #for i in range(0,len(preds)):
        #    sentence_acc.append(int(all(preds[i] == test_tags_y[i])))
        #print("Test Set Sentence Accuracy: ",np.sum(sentence_acc)/len(sentence_acc))
    else:
        print("Info: The script assumes 1 argument for file path for the test file.")
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

def to_categorical(sequences, categories):
        cat_sequences = []
        for s in sequences:
            cats = []
            for item in s:
                cats.append(np.zeros(categories))
                cats[-1][item] = 1.0
            cat_sequences.append(cats)   
        return np.array(cat_sequences)
def load_model():
        model = tf.keras.models.load_model('models/POS_Tagger',custom_objects={'custom_accuracy':ignore_pad_accuracy(0)},)
        return model
def ignore_pad_accuracy(to_ignore=0):
        def custom_accuracy(y_true, y_pred):
            y_true_class = K.argmax(y_true, axis=-1)
            y_pred_class = K.argmax(y_pred, axis=-1)
    
            ignore_mask = K.cast(K.not_equal(y_pred_class, to_ignore), 'int32')
            matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
            accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
            return accuracy
        return custom_accuracy

def build_generate_text(test_samples,word_to_ix,MAX_LENGTH):
    test_samples_X = []
    for s in test_samples:
        s_int = []
        for w in s:
            try:
                s_int.append(word_to_ix[w.lower()])
            except KeyError:
                s_int.append(word_to_ix['-OOV-'])
        test_samples_X.append(s_int)
    test_samples_X = pad_sequences(test_samples_X, maxlen=MAX_LENGTH, padding='post')
    return test_samples_X

if __name__ == "__main__":
    if len(sys.argv) ==2:
        #print(sys.argv)
        text_file = sys.argv[1]
        with open(text_file) as file:
            text = file.read()
        text_lines = text.split(".")
        if text_lines[-1] == "":
            #print("Text has empty token",len(text_lines[-1]))
            text_lines = text_lines[:-1]

        train_vocab_X = joblib.load("vocab/word_to_idx")
        train_vocab_y = joblib.load("vocab/tag_to_idx")
        MAX_LENGTH = joblib.load("vocab/Train_max_len")

        
        text_tokens = [text.split() for text in text_lines]
        #print("Text tokens: ",text_tokens[:2])

        test_sentences_X = build_generate_text(text_tokens,train_vocab_X,MAX_LENGTH)
        #print("Sentences: ",test_sentences_X[0])
        pos = load_model()
        predictions = pos.predict(test_sentences_X)
        abs_preds = np.apply_along_axis(np.argmax,2,predictions)
        #print(abs_preds.shape)
        #print(abs_preds[:2])
        pred_tags_doc = []
        for i, pred in enumerate(abs_preds):
            pred_tags_line = []
            pred_nums = pred[:len(text_tokens[i])]
            #print(pred_nums)
            #print(train_vocab_y.items())
            for val in pred_nums:
                for key,value in train_vocab_y.items():
                    if value==val:
                        if val == 0:
                            pred_tags_line.append("X")
                        else:
                            pred_tags_line.append(key)
            pred_tags_doc.append(pred_tags_line)
        print("\n****POS Tagging Output:*****\n")
        for i,line in enumerate(text_lines):
            print("Input Sentence: ",line)
            print("POS Tags: ",pred_tags_doc[i])
    else:
        print("Info: The script assumes 1 argument for file path for the test file.")
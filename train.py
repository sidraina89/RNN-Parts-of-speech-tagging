import dataload, calculate_embeddings
import sys
import POSTagger
import conllu
#import sklearn.externals.joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib


def token_to_idx(data,word_or_tag="WORD"):
    tok_to_idx = dict()
    pos = int(word_or_tag=="TAG")

    tok_to_idx["-PAD-"] = 0
    if word_or_tag == "WORD":
        tok_to_idx["-OOV-"] = 1
    
    for i in range(len(data)):
        seq = data[i][pos]
        for tok in seq:
            if pos == 0:
                tok = tok.lower()
            if tok not in tok_to_idx:
                tok_to_idx[tok] = len(tok_to_idx)
    return tok_to_idx

def convert_token_to_number(df_list,vocab,word_or_tag="WORD"):
    token_ids = []
    pos = int(word_or_tag=="TAG")
    #print("Passed: ",df_list[0][pos])
    #print(vocab)
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

def build_train_dev_test_data(train_file=None,dev_file=None,test_file=None):
    raw_data_list = dataload.load_raw_conllu(train_file,dev_file)
    train_data,dev_data = dataload.convert2df(raw_data_list)
    # Build vocabulary from words and tags in training set
    train_df_list = [[d["form"].tolist(), d["upos"].tolist()] for d in train_data]
    dev_df_list = [[d["form"].tolist(), d["upos"].tolist()] for d in dev_data]
    train_vocab_X = token_to_idx(train_df_list)
    train_vocab_y = token_to_idx(train_df_list,"TAG")
    #print(train_vocab_y.items())

    train_sentences_X = convert_token_to_number(train_df_list,train_vocab_X,"WORD")
    train_tags_y = convert_token_to_number(train_df_list,train_vocab_y,"TAG")

    dev_sentences_X = convert_token_to_number(dev_df_list,train_vocab_X,"WORD")
    dev_tags_y = convert_token_to_number(dev_df_list,train_vocab_y,"TAG")
    
    return train_sentences_X,dev_sentences_X,train_tags_y,dev_tags_y,train_vocab_X,train_vocab_y

def pad_sequence_list(seq_list,MAX_LENGTH):
    return [pad_sequences(seq, maxlen=MAX_LENGTH, padding='post') for seq in seq_list]


if __name__ == "__main__":
    if len(sys.argv) ==3:
        #print(sys.argv)
        train_file = sys.argv[1]
        dev_file = sys.argv[2]
        train_sentences_X,dev_sentences_X,train_tags_y,dev_tags_y,train_vocab_words,train_vocab_y = build_train_dev_test_data(train_file,dev_file)
        #print(train_sentences_X)
        # Finding max number of tokens in a sequence. (The sentence with the most words.)
        MAX_LENGTH = len(max(train_sentences_X, key=len))
        # Padding all sequences to make them equal - Both words and Tags
        train_sentences_X,dev_sentences_X,train_tags_y,dev_tags_y = pad_sequence_list([train_sentences_X,dev_sentences_X,train_tags_y,dev_tags_y],MAX_LENGTH)
        embedding_dim = 300
        #embedding_matrix = calculate_embeddings.calculate_embeddings(train_vocab_words,embedding_dim)
        pos = POSTagger.POSTagger(embedding_dim,train_vocab_words,train_vocab_y,MAX_LENGTH)
        pos_model = pos.train_model(train_sentences_X,train_tags_y,dev_sentences_X,dev_tags_y,50)
        pos.show_metrics(dev_sentences_X,dev_tags_y)
        # Save model
        pos.save_model()
        joblib.dump(train_vocab_words,"vocab/word_to_idx")
        joblib.dump(train_vocab_y,"vocab/tag_to_idx")
        joblib.dump(MAX_LENGTH,"vocab/Train_max_len")
        print("Training vocabulary saved at - vocab/word_to_idx")
    else:
        print("Info: The script assumes 2 arguments for file paths for the train and dev files. Please run as follows:\n python3 train.py [train file path] [dev file path]!")


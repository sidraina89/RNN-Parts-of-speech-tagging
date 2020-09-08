import os
import numpy as np
def calculate_embeddings(word_to_ix,embedding_dim):
    #Pre-Trained Embeddings from Word-2-Vec - Google News 300d 
    EMBEDDING_DIM = embedding_dim
    GLOVE_DIR = "glove/glove.6B"
    embeddings_index = {}
    print("Loading Pre_trained Embeddings")
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.{}d.emb'.format(EMBEDDING_DIM)))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print("Loaded Pre_trained Embeddings")

    embedding_matrix = np.zeros((len(word_to_ix) + 1, EMBEDDING_DIM))
    for word, i in word_to_ix.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector # words not found in embedding index will be all-zeros
    return embedding_matrix

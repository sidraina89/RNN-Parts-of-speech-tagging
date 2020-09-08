# POS Tagging ML Model
This work is to develop a part-of-speech(POS) tagger based on a Bi-directional RNN (GRU cell) Model trained on the Georgetown University Multilayer(GUM) Corpus.
Link to data: (https://github.com/UniversalDependencies/UD_English-GUM/tree/master).
### POS Tag Sets
We will use the part-of-speech tags used in the Universal Dependencies Project:

|TAG|POS|
|---|---|
|ADJ|adjective|
|ADP|adposition|
|ADV|adverb|
|CCONJ|coordinating conjunction|
|DET|determiner|
|INTJ|interjection|
|NOUN|noun|
|NUM|numeral|
|PART|particle|
|PRON|pronoun|
|PROPN|proper noun|
|PUNCT|subordinating conjunction|
|SYM|symbol|
|VERB|verb|
|X|other|

--------------------------------
#### Assumptions Made:
1. Train, dev and test set are mutually exclusive and POS token distribution is similar among the data sets.
2. Model accuracy is measured per token i.e. *sum(No. of POS tokens correctly predicted)/Total POS Tokens in document*
3. POS Tagging is assumed to be a sequence modeling task, i.e. the POS token classes depend upon the tokens before and after them in the document.

#### Model Choice:
As I assumed POS Tagging to be a sequence modeling task, I chose to use a Bidirectional RNN (GRU cell) to model this task. Bidirectional RNNs give an ability to receive information from both past and future states.Thus input data from the past and future of the current time frame can be used to calculate the same output giving us an ability to take the tokens before and after a current token into consideration, thus saving time to build hand crafted features to do the same.

#### Testing Strategy:
The model has been trained on the training set from the GUM corpus and tuned for accuracy on the dev set from GUM.
As a Bidirectional LTSM takes in input sequences of the same length, the documents in the data sets were post padded with a custom token "-PAD-" to make them equal in length.
e.g.

##### Tokens before padding:

```python
-tok1- -tok2- -tok3-
-tok1- -tok2- -tok3- -tok4- -tok5- -tok6-
```
##### Tokens after padding:

```python
-tok1- -tok2- -tok3- -PAD- -PAD- -PAD-
-tok1- -tok2- -tok3- -tok4- -tok5- -tok6-
```

Thus when the model predicts on a new padded sequence, the accuracy also takes into account the custom "-PAD-" tokens which creates an overly optimistic view of the performance of the model as the -PAD- token is pretty easy to predict as they usually are surrrounded by other -PAD- tokens.
In order to overcome this, a custom accuracy metric has been used to train the model which ignores the matches of the custom -PAD- token.

This custom accuracy metric is defined as the ```ignore_pad_accuracy``` function in ```class POSTagger```.

### Results:

Accuracy on Dev set of tokens is 90.31% (custom accuracy metric)

Classification Report from the dev data set for every POS tag:

|POS Tag|precision|recall|f1-score|support|
|-------|---------|------|--------|-------|
|ADJ|0.84|0.77|0.81|1113|
|NOUN|0.78|0.90|0.84|2885|
|CCONJ|0.98|1.00|0.99|525|
|PUNCT|1.00|1.00|1.00|2023|
|ADP|0.94|0.97|0.96|1660|
|PROPN|0.64|0.54|0.59|951|
|SCONJ|0.89|0.74|0.81|311|
|AUX|0.97|0.97|0.97|718|
|VERB|0.92|0.85|0.88|1639|
|DET|0.99|0.99|0.99|1325|
|PRON|0.97|0.97|0.97|1119|
|NUM|0.95|0.81|0.88|337|
|ADV|0.87|0.80|0.84|557|
|X|0.17|0.27|0.21|30|
|SYM|1.00|0.90|0.95|10|
|PART|0.95|0.92|0.94|371|
|INTJ|1.00|0.63|0.77|19|

##### Time Spent: 
14-18 hrs (apx)

##### How to use:
This module requires python3 and the dependency packages are listed down in ```requirements.txt```
The dependencies can be downloaded by running ```pip3 install -r requirements.txt```
After downloading dependencies:
1. Train the model by executing ```train.py``` and pass the train and dev file paths as arguments. e.g.:
    ```python3 train.py data/en_gum-ud-train.conllu data/en_gum-ud-dev.conllu```
2. Evaluate on test data by executing ```eval.py``` and passing the test set file path as an argument. e.g.:
    ```python3 evaluate.py data/en_gum-ud-test.conllu```
3. Generate POS tokens for unlabeled text file by executing ```generate.py``` and passing the unlabeled text file path as an argument. e.g.:
    ```python3 generate.py unlabelled_text.txt```

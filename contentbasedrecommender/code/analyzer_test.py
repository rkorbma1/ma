import pandas as pd
import json
from sklearn.decomposition import TruncatedSVD
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from textblob_de import TextBlobDE as TextBlob
from textblob_de.lemmatizers import PatternParserLemmatizer
import spacy
nlp = spacy.load('de_core_news_md')

dataframe = pd.read_csv('titatest.csv')

print(len(dataframe))

dataframe.dropna(subset=['article_trans_teaser'], inplace=True)

"""
for index, row in dataframe.iterrows():
    print(index)
    ind = dataframe.audio_id[index]
    url = dataframe.audio_path_abs[index]
    doc = requests.get(url)
    name = './audio/' + str(ind)
    print(name)
    with open(name , 'wb') as f:
       f.write(doc.content)
"""
#[769, 806, 458, 199, 94, 93, 95, 785, 200, 202, 191, 334, 808, 96, 203, 97, 98, 273, 99, 608, 206, 101, 102, 210, 399, 105, 106, 740, 386, 802, 107, 782, 788, 492, 215, 217, 219, 841, 109, 737, 630, 731, 741, 743, 732, 733, 111, 734, 292, 738, 112, 113, 222, 223, 225, 116, 807, 117, 233, 119, 605, 121, 122, 728, 727, 123, 124, 17, 125, 239, 694, 254, 243, 126, 127, 128, 129, 130, 247, 131, 249, 132, 134, 135, 253, 136, 258, 261, 142, 265, 143, 607, 144, 266, 145, 684, 272, 146, 147, 274, 275, 340, 822, 278, 148, 280, 152, 281, 154, 649, 283, 158, 780, 523, 541, 115, 288, 444, 167, 618, 169, 271, 264, 821, 177, 844, 259, 180, 182, 810, 610, 16, 248, 184, 186, 453, 614, 613, 185, 751, 192, 290, 195, 198, 423, 197, 196, 194, 774, 600, 778, 617, 849, 840, 789, 238, 171, 234, 759, 168, 729, 165, 611, 830, 229, 163, 162, 755, 161, 155, 11, 218, 216, 799, 211, 160, 150, 149]

stop_words =stopwords.words('german')

print('Ich lade gleich das Model! ')

#model = KeyedVectors.load_word2vec_format('dewiki_20180420_300d.txt.bz2', binary=False)
model = KeyedVectors.load('word2vec_german.model')
#model.save("word2vec_german.model")

print('Model ist geladen!')

data = dataframe.article_trans_teaser

data.dropna(inplace=True)

print('Schritt 1.')

def preprocess(doc):
    doc = [ent.lemma_ for ent in nlp(doc)]
    doc = [w.lower() for w in doc]
    doc = [w for w in doc if not w in stop_words]  # Remove stopwords.
    doc = [w for w in doc if w in model.vocab]  # Remove words which are not in the vocabulary.
    doc = [w for w in doc if w.isalpha() and len(w)>1] # Remove numbers and punctuation.
    return doc

def preprocess_similar_words(doc):
    doc = [ent.lemma_ for ent in nlp(doc) if ent.pos_ == 'NOUN' or ent.pos_ == 'PROPN']
    doc = ' '.join(doc)
    doc = [ent.lemma_ for ent in nlp(doc) if ent.pos_ == 'NOUN' or ent.pos_ == 'PROPN']
    doc = [w for w in doc if not w in stop_words] # Remove stopwords.
    doc = [w.lower() for w in doc]
    doc = [w for w in doc if w in model.vocab]  # Remove words which are not in the vocabulary.
    doc = [w for w in doc if w.isalpha() and len(w)>1] # Remove numbers and punctuation.
    return doc

wmd_corpus = []
for v in data:
    wmd_corpus.append(preprocess(v))

print(wmd_corpus)
print(len(wmd_corpus))

prep = []

for values in wmd_corpus:
    prep.append(json.dumps(values))

dataframe['preprocess'] = prep

preprocess_sim=[]

for v in data:
    preprocess_sim.append(preprocess_similar_words(v))

print(preprocess_sim)
print(len(preprocess_sim))

prep_nouns = []

for values in preprocess_sim:
    prep_nouns.append(json.dumps(values))

dataframe['nouns'] = prep_nouns

print('Schritt 2.')
import numpy as np

REAL = np.float32

def sif_embeddings(sentences, model, alpha=1e-3):
    """Compute the SIF embeddings for a list of sentences
    Parameters
    ----------
    sentences : list
        The sentences to compute the embeddings for
    model : `~gensim.models.base_any2vec.BaseAny2VecModel`
        A gensim model that contains the word vectors and the vocabulary
    alpha : float, optional
        Parameter which is used to weigh each individual word based on its probability p(w).
    Returns
    -------
    numpy.ndarray
        SIF sentence embedding matrix of dim len(sentences) * dimension
    """

    vlookup = model.wv.vocab  # Gives us access to word index and count
    vectors = model.wv  # Gives us access to word vectors
    size = model.vector_size  # Embedding size

    Z = 0
    for k in vlookup:
        Z += vlookup[k].count  # Compute the normalization constant Z

    output = []

    # Iterate all sentences
    for s in sentences:
        count = 0
        v = np.zeros(size, dtype=REAL)  # Summary vector
        # Iterare all words
        for w in s:
            # A word must be present in the vocabulary
            for w in s:
                if w in vlookup:
                    # The loop over the the vector dimensions is completely unecessary and extremely slow
                    v += (alpha / (alpha + (vlookup[w].count / Z))) * vectors[w]
                    count += 1
        if count > 0:
            for i in range(size):
                v[i] *= 1 / count
        output.append(v)
    return np.vstack(output).astype(REAL)

def remove_first_principal_component(X):
    svd = TruncatedSVD(n_components=1, n_iter=20, random_state=0)
    svd.fit(X)
    pc = svd.components_
    XX = X - X.dot(pc.transpose()) * pc
    return XX

wmd_corpus = [json.loads(w) for w in dataframe.preprocess]

print(wmd_corpus)

result = sif_embeddings(wmd_corpus, model)

embeddings = remove_first_principal_component(result)

print(embeddings)

print(type(embeddings))

print('Schritt 3.')

vectors=[]

for values in embeddings:
    vectors.append(json.dumps(values.tolist()))

dataframe['vectors'] = vectors

print(dataframe.columns)

dataframe.to_csv('titatest.csv')

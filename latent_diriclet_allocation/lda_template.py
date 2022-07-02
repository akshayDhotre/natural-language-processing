
import nltk
import pandas as pd
import re
import string
from pprint import pprint
from pip import main
from wordcloud import WordCloud
import gensim
import pyLDAvis.gensim_models
import pickle 
import pyLDAvis
import os

from nltk import PorterStemmer
from nltk import WordNetLemmatizer

pd.set_option('display.max_colwidth', 100)
stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(['model'])


# clean text by removing punctuations and stopwords
def clean_text(text):
    text = ''.join([word for word in text if word not in string.punctuation])
    tokens = re.split(r'\W+', text.lower())
    text = [word for word in tokens if word not in stopwords and len(word) > 3]
    return text


# stemming of given tokens
def stemming_text(tokenized_text):
    ps = PorterStemmer()
    text = [ps.stem(word) for word in tokenized_text]
    return text


# lemmetization of given tokens
def lemmetizing_text(tokenized_text):
    wn = WordNetLemmatizer()
    text = [wn.lemmatize(word) for word in tokenized_text]
    return text


# visualize wordnet from given list of sentences
def visualize_word_cloud(document_list):
    long_string = ','.join(list(document_list))
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
    wordcloud.generate(long_string)
    wordcloud.to_image().show()


# create bag of word for text
def create_bow_corpus(data_list):
    dictionary = gensim.corpora.Dictionary(data_list)
    corpus = [dictionary.doc2bow(text) for text in data_list]
    return corpus, dictionary


# create lda model
def create_lda_model(corpus, dictionary, num_topics):
    lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=num_topics)
    pprint(lda_model.print_topics())
    return lda_model


# visualize lda topics
def visualize_topics_pyldavis(lda_model, corpus, id2word, num_topics):
    LDAvis_data_filepath = os.path.join('lda_vis_files\ldavis_prepared_'+str(num_topics))

    # # this is a bit time consuming - make the if statement True
    # # if you want to execute visualization prep yourself
    if 1 == 1:
        LDAvis_prepared = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
        with open(LDAvis_data_filepath, 'wb') as f:
            pickle.dump(LDAvis_prepared, f)
    # load the pre-prepared pyLDAvis data from disk
    with open(LDAvis_data_filepath, 'rb') as f:
        LDAvis_prepared = pickle.load(f)
    pyLDAvis.save_html(LDAvis_prepared, 'lda_vis_files\ldavis_prepared_'+ str(num_topics) +'.html')
    return LDAvis_prepared


if __name__ == '__main__':
    
    # read the data
    papers = pd.read_csv('data/papers.csv')
    # keep only impotant columns
    papers = papers.drop(['id', 'event_type', 'pdf_name', 'year', 'title', 'abstract'], axis=1).sample(100)
    # clean text
    papers['cleaned_text'] = papers['paper_text'].apply(lambda x: clean_text(x))
    # lemmatize text
    papers['cleaned_text_lemma'] = papers['cleaned_text'].apply(lambda x: lemmetizing_text(x))
    # create corpus with bag of words
    corpus, dictonary = create_bow_corpus(papers['cleaned_text_lemma'])
    # create lda model
    lda_model = create_lda_model(corpus, dictonary, num_topics=10)
    # visualize topic
    plda = visualize_topics_pyldavis(lda_model, corpus, dictonary, num_topics=10)

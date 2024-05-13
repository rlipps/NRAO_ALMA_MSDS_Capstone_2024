# Packages
import pandas as pd
import numpy as np
import nltk
import re
import string
from joblib import load
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# NRAO Stopword list
nrao_stops = ['0','1','2','3','4','5','6','7','8','9',"=",'<','>','~',"/",'`',".",",", 'observation'
                       'alma','resolution','source','show','sample','high','use','observe'
                       'figure','low','image','propose', 'also','use', 'large', 'study'
                       'reference', 'detect','see','well', 'time', 'however', 'expect'
                       'provide','datum', 'model', 'result','sensitivity','scale'
                       'find','allow','scientific','target','compare','resolve','first',
                       'leave', 'estimate','suggest', 'due', 'obtain', 'small', 'measure',
                       'include','property','justification', 'right ', 'understand', 'similar',
                       'detection', 'require', 'indicate', 'order', 'range', 'make','map','thus',
                       'follow','fig','goal','proposal','field','determine','therefore', 'reveal',
                       'give', 'process','total', 'important', 'know','constrain', 'ratio','even',
                       'case','et', 'al', 'pc','kpc','apj','km','mm','m','one','two', 'data', 'us',
                       'mnras', 'left', 'right', 'may', 'within','would','need','request','mjy',
                       'different','assume','recent','good','since','still','previous','science',
                       'ghz','could','object','much','survey','three','whether','likely','several',
                       'like','able','identify','new','best','number','analysis','confirm','predict',
                       'le','evidence','select','example','take','recently','combine','exist','value',
                       'fit','objective','comparison','investigate','respectively','many','although',
                       'achieve','cm','jy','need','enough','search','yr','explain','au','apjl','per','arxiv',
                       'a&a', 'aa', 'apj', 'apjl', 'mnras', 'pasp', 'aj', 'cycle', 'band',
                       'emission', 'free', 'anticipate', 'originate', 'success', 'separate', 'uv', 'significance',
                       'hot', 'frequency', 'wavelength', 'realistic', 'mas', 'mg', 'minute', 'ii', 'ad', 'hd',
                       'occurrence', 'event', 'myr', 'ra', 'dec',  'ly', 'tau', 'cn',
                       'arc', 'ori', 'hh', 'iii', 'cha', 'ab', 'tw', 'ms', 'ngc', 'pds',
                       'jwst','hcn','hco+','oh','xray','aca', 'vla', 'gbt', 'proto', 'noema',
                       'quiescent','nir', 'heating', 'sb','temperature','cr','hya','liu','warm',
                       'nh','extent','spitzer', 'co','yang']

# Text processing functions
#convert to lowercase, strip and remove punctuations and remove ALMA, case insensitive
def preprocess(text):
    text = text.lower()                                                         # Make everything lower case
    text = text.strip()                                                         # Strip leading and trailing whitespace
    text = re.compile('<.*?>').sub('', text)                                    # Remove things like html tags 
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)    # Remove punctuation
    text = re.sub(r'(?i)alma', '', text)                                        # Remove case insensitive 'alma'
    text = re.sub('\s+', ' ', text)                                             # Convert whitespace to single space
    text = re.sub(r'\[[0-9]*\]',' ',text)                                       # Remove things like citations e.g. [9]
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())                    # Remove non alphanumeric characters
    text = re.sub(r'\d',' ',text)                                               # Remove digits
    text = re.sub(r'\s+',' ',text)                                              # Collapse any created whitespace into single space
    return text
 
# STOPWORD REMOVAL
def stopword(string):
    sws = stopwords.words('english')
    sws.extend(nrao_stops)
    a= [i for i in string.split() if i not in sws]
    return ' '.join(a)

#LEMMATIZATION
# Initialize the lemmatizer
wl = WordNetLemmatizer()
 
# This is a helper function to map NTLK position tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatizer(string):
    word_pos_tags = nltk.pos_tag(word_tokenize(string)) # Get position tags
    a=[wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token
    return " ".join(a)

# Import models from joblib files
logreg_tfidf_vectorizer = load('models/line_continuum_classifier/tfidf_vectorizer_logreg.joblib')
logreg_classifier = load('models/line_continuum_classifier/log_model.joblib')
naive_bayes_tfidf_vectorizer = load('models/band_classification/tfidf_vectorizer_naive_bayes.joblib')
naive_bayes_model = load('models/band_classification/naive_bayes_model.joblib')
lda_count_vectorizer = load('models/topic_mining/lda_count_vectorizer.joblib')
lda_model = load('models/topic_mining/lda_model.joblib')

# Prompt user for title and abstract
title = input('Enter project title:')
abstract = input('Enter project abstract:')
raw_text = title + '. ' + abstract

# Create dataframe of input with various text processing required for models
df = pd.DataFrame({
    raw_text
}, index=[1])
df['std_text'] = df.raw_text.apply(lambda x: preprocess(x))
df['std_text_sw_removed'] = df.std_text.apply(lambda x: stopword(x))
df['std_text_sw_removed_lemmatized'] = df.std_text_sw_removed.apply(lambda x: lemmatizer(x))

# Predict from models
# Logistic Regression for Line/Continuum classification
logreg_pred = logreg_classifier.predict(logreg_tfidf_vectorizer(df.std_text))
print(f'Predicted probability of only continuum measurements: {round(logreg_pred[0][0]*100, 3)}')
print(f'Predicted probability of at least one line measurement: {round(logreg_pred[0][1]*100, 3)}')

# Topic assignment
lda_pred = np.argmax(lda_model.transform(lda_count_vectorizer(df.std_text_sw_removed_lemmatized)))
print(f'Predicted project topic number: {lda_pred}')

# Band prediction

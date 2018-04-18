from sklearn.base import TransformerMixin, BaseEstimator
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, ENGLISH_STOP_WORDS
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.stem.wordnet import WordNetLemmatizer
#from nltk.stem.snowball import SnowballStemmer
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
#from sklearn.linear_model import LogisticRegression
import pickle


class InquireBoulderData(object):
    '''
    Reads data from Inquire Boulder formatted files
    '''
    def __init__():
        self.categories = {}
        self.categories_reversed = {}
        self.X = None
        self.y = None
        #self.data_filename= data_filename
        #self.topic_filename = topic_filename

    def fit(self,data_filename=None,topic_filename=None)
        if data_filename == None:
            data_filename = 'data/Inquire_Boulder_All_requests_2017_with_details.csv'
        data_df = pd.read_csv(data_filename)

        if topic_filename == None:
            topic_filename = 'data/topics.csv'
        topics_df = pd.read_csv(topic_filename,sep=';')

        df_c2 = data_df.set_index('Topic').join(topic_df.set_index('subcategory'))
        df_c2 = df_c2.reset_index()
        cats = df_c2['category'].unique()
        for i, cat in enumerate(cats):
            self.categories_reversed[cat]=i
            self.categories[i]=cat
        df_c2['cat_num']=df_c2['category'].map(self.categories_reversed)
        self.X = df_c2.Description
        self.y = df_c2['cat_num']
        return self,X,y

class EmailData(object):
    '''
    Reads data from Boulder City Council Email formatted files
    '''
    def __init__():
        self.X = None

    def fit(self,data_filename=None):
        if data_filename == None:
            data_filename = 'data/jan_mar_2018.csv'
        data_df = pd.read_csv(data_filename)
        mail_df = mail_df[mail_df['Body'].isnull()==False]
        self.X = mail_df['Body']

        return self,self.X


class LemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        lemma = WordNetLemmatizer()
        analyzer = super(LemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([lemma.lemmatize(w) for w in analyzer(doc)])



class TopicClassifier(object):
    def __init__():
        self.model=None
        self.categories = {}

    def fit(self,X,y):
        pipeline = Pipeline([
            ('vect', LemmedCountVectorizer(stop_words='english',analyzer="word",max_df=0.5)),
            ('tfidf', TfidfTransformer(norm='l2')),
            ('clf', SGDClassifier(penalty='elasticnet',alpha=0.0001,loss='log',class_weight='balanced')),
        ])
        pipeline.fit(X,y)
        self.model=pipeline
        return self

    def predict(X):
        y_pred = model.predict(X)
        y_cats = np.vectorize(self.categories.get)(y_pred)
        return y_cats,y_pred

    def save_model_to_pickle(filename=None):
        if filename == None:
            filename = 'models/pipeline_to_SGD_logloss.p'
        pickle.dump(pipeline,open(filename,'wb'))

    def load_model_from_pickle(filename=None):
        if filename == None:
            filename = 'models/pipeline_to_SGD_logloss.p'
        self.model = pickle.load(open(filename,'rb'))

    def load_categories_from_pickle(filename=None):
        if filename == None:
            filename = 'data/categories_with_index.p'
        self.categories = pickle.load(open(filename, 'rb'))

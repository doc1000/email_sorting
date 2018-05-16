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
#import models.mylemmatizer
# from models.mylemmatizer import LemmedCountVectorizer


class LemmedCountVectorizer(CountVectorizer):
    #def __init__(self):
    #    return self

    def build_analyzer(self):
        lemma = WordNetLemmatizer()
        analyzer = super(LemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([lemma.lemmatize(w) for w in analyzer(doc)])


class InquireBoulderData(object):
    '''
    Reads data from Inquire Boulder formatted files
    '''
    def __init__(self):
        self.categories = {}
        self.categories_reversed = {}
        self.X = None
        self.y = None

        #self.data_filename= data_filename
        #self.topic_filename = topic_filename

    def fit(self,data_filename=None,topic_filename=None, target='topic'):
        if data_filename == None:
            data_filename = 'data/Inquire_Boulder_All_requests_2017_with_details.csv'
        data_df = pd.read_csv(data_filename)
        self.X = data_df.Description
        self.X = text_cleaning(self.X)
        if target=='topic':
            if topic_filename == None:
                topic_filename = 'data/topics.csv'
            topics_df = pd.read_csv(topic_filename,sep=';')

            df_c2 = data_df.set_index('Topic').join(topics_df.set_index('subcategory'))
            df_c2 = df_c2.reset_index()
            cats = df_c2['category'].unique()
            for i, cat in enumerate(cats):
                self.categories_reversed[cat]=i
                self.categories[i]=cat
            df_c2['cat_num']=df_c2['category'].map(self.categories_reversed)
            self.y = df_c2['cat_num']
        if target =='recipient':
            self.y = data_df['Assigned To']
        return self.X,self.y

class EmailData(object):
    '''
    Reads data from Boulder City Council Email formatted files
    '''
    def __init__(self):
        self.X = None

    def fit(self,data_filename=None):
        if data_filename == None:
            data_filename = 'data/jan_mar_2018.csv'
        mail_df = pd.read_csv(data_filename)
        mail_df = mail_df[mail_df['Body'].isnull()==False]
        self.X = mail_df['Body']

        return self.X




class TopicClassifier(object):
    def __init__(self):
        self._model=None
        self.categories = {}
        self.X = None
        self._vect = None
        self._tfidf_matrix = None
        self.vocabulary_ = None
        #from topic_classifier import LemmedCountVectorizer

    def fit(self,X,y):
        X = self.text_cleaning(X)
        pipeline = Pipeline([
            ('vect', LemmedCountVectorizer(stop_words='english',analyzer="word",max_df=0.5)),
            ('tfidf', TfidfTransformer(norm='l2')),
            ('clf', SGDClassifier(penalty='elasticnet',alpha=0.0001,loss='log',class_weight='balanced')),
        ])
        pipeline.fit(X,y)
        self._model=pipeline
        self.vocabulary_ = self._model.named_steps['vect'].vocabulary_
        self._vect = model._model.named_steps['vect'].transform(X)
        self._tfidf_matrix = model._model.named_steps['tfidf'].transform(self._vect)
        return self

    def predict(self,X,return_type=1):
        X = self.text_cleaning(X)
        self.X = X
        y_pred = self._model.predict(X)
        y_cats = np.vectorize(self.categories.get)(y_pred)
        # if return_type == 1:
        #     y_return = y_cats
        # else:
        #     y_return = y_pred

        return y_cats


    def save_model_to_pickle(self,filename=None):
        if filename == None:
            filename = 'models/pipeline_to_SGD_logloss.p'
        pickle.dump(self._modelopen(filename,'wb'))

    def load_model_from_pickle(self,filename=None):
        if filename == None:
            filename = 'models/pipeline_to_SGD_logloss.p'
        self._model = pickle.load(open(filename,'rb'))
        self.load_categories_from_pickle()

    def load_categories_from_pickle(self,filename=None):
        if filename == None:
            filename = 'data/categories_with_index.p'
        self.categories = pickle.load(open(filename, 'rb'))

    def save_predictions_to_csv(self,filename=None):
        if filename == None:
            filename = 'data/predicted_category_log.csv'
        mail_df.to_csv(filename)

    def text_cleaning(self,docs):
        import re
        url_reg  = r'[a-z]*[:.]+\S+'
        for i,source_text in enumerate(docs):
            result = re.sub(url_reg, '', source_text)
            result = re.sub('_x000D_',' ',result)
            docs[i]=result
        return docs

    def top_words(self,doc):
        doc = [doc]
        #vect_mod = self._model.named_steps['vect']
        vect = self._model.named_steps['vect'].transform(doc)
        rev_d = {v:k for k,v in self.vocabulary_.items()}
        index_d = 0
        one_doc = vect[index_d].toarray()
        #one_doc = tf_matrix[index_d].toarray()
        top_words = [rev_d[x] for x in np.argsort(one_doc*-1)[0][:20] if len(rev_d[x])>4 and one_doc[0,x]>0]
        return top_words

    def similar_emails(self,doc):
        from sklearn.metrics.pairwise import cosine_similarity
        doc = [doc]
        vect = model._model.named_steps['vect'].transform(doc)
        tf_matrix = model._model.named_steps['tfidf'].transform(vect)
        #vect_X = model._model.named_steps['vect'].transform(model.X)
        #tf_matrix_X = model._model.named_steps['tfidf'].transform(vect_X)

        #index_d = 0
        cossim = cosine_similarity(tf_matrix[:],self._tfidf_matrix[:])
        #cossim[0][index_d] = 0
        comp_docs = [(cossim.reshape(-1,1)[i][0],model.X[i]) for i in np.argsort(cossim*-1)[0][:5]]
        print(comp_docs[0])


def text_cleaning(docs):
    import re
    url_reg  = r'[a-z]*[:.]+\S+'
    for i,source_text in enumerate(docs):
        result = re.sub(url_reg, '', source_text)
        result = re.sub('_x000D_',' ',result)
        docs[i]=result
    return docs

def load_up_model():
    X_train,y_train = InquireBoulderData().fit()
    model = TopicClassifier()
    #model.fit(X_train,y_train)
    model.load_model_from_pickle()
    X = EmailData().fit()
    y = model.predict(X)

    df = pd.DataFrame(model.X)
    df['topic']=y
    #df.to_csv('data/results.csv')
    return model, df



if __name__ == '__main__':
    #similar messages
    from sklearn.metrics.pairwise import cosine_similarity
    model, _ = load_up_model()
    doc = 'the city counsel needs to address ausault weapon control.  guns are killing our kids'
    doc = [doc]
    vect = model._model.named_steps['vect'].transform(doc)
    tf_matrix = model._model.named_steps['tfidf'].transform(vect)
    vect_X = model._model.named_steps['vect'].transform(model.X)
    tf_matrix_X = model._model.named_steps['tfidf'].transform(vect_X)

    index_d = 0
    cossim = cosine_similarity(tf_matrix[:],tf_matrix_X[:])
    #cossim[0][index_d] = 0
    comp_docs = [(cossim.reshape(-1,1)[i][0],model.X[i]) for i in np.argsort(cossim*-1)[0][:5]]
    print(comp_docs[0])
    #i might need to append any unknown words to the existing tfidf.  maybe
    #or could just transform doc directly using named steps... der... already fit.

#mail_df['predicted_category']=y_cats
#mail_df.to_csv('data/predicted_category_log.csv')

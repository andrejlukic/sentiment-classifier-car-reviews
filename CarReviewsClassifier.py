import pandas as pd
import time
import matplotlib as plt
import nltk
import ssl
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

english_stemmer = nltk.stem.SnowballStemmer('english')
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([english_stemmer.stem(w) for w in analyzer(doc)])

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: ([english_stemmer.stem(w) for w in analyzer(doc)])

class CarReviewsClassifier():

    def __init__(self, fpath):

        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        nltk.download('stopwords')
        nltk.download('punkt')

        dataset = pd.read_csv(fpath)
        print("Loaded {} car reviews".format(dataset.shape[0]))
        self.trainsetX, self.testsetX, self.trainsetY, self.testsetY = \
            train_test_split(dataset.Review, dataset.Sentiment, test_size=0.2)
                             #, random_state=44)


    def train_and_test(self, stemm=True, vectorizer='count', min_df=2, max_df=1.0, ngram_range=(1,2), binary=True):


        count_vect = StemmedTfidfVectorizer(analyzer="word", stop_words='english', min_df=min_df, max_df=max_df, ngram_range=ngram_range, binary=binary, lowercase=True)
        classifier = LinearSVC()

        self.pipeline = Pipeline([
            ('vectorizer', count_vect),
            ('classifier', classifier)
        ])

        # call fit as you would on any classifier
        self.pipeline.fit(self.trainsetX, self.trainsetY)

        # predict test instances
        print(self.pipeline.score(self.testsetX, self.testsetY))
        return self.pipeline.score(self.testsetX, self.testsetY)

    def grid_search_vectorizer_params(self):

        classifier = LinearSVC()
        count_vect = TfidfVectorizer(stop_words='english')

        self.pipeline = Pipeline([
            ('vectorizer', count_vect),
            ('classifier', classifier)
        ])

        print(self.pipeline.get_params().keys())

        parameters = [{
            'classifier': (LinearSVC(),),
            # 'classifier__alpha': (0.9,),
            'vectorizer__binary': (True,),
            'vectorizer__lowercase': (True,),
            'vectorizer__max_df': (1.0,),
            'vectorizer__min_df': (2,3,4),
            'vectorizer__ngram_range': ((1,2),(1,1)),
            'vectorizer': (StemmedTfidfVectorizer(),)
        }]

        grid_search = GridSearchCV(self.pipeline, parameters, verbose = 3, n_jobs = -1)
        clf = grid_search.fit(self.trainsetX, self.trainsetY)
        score = clf.score(self.testsetX, self.testsetY)
        print("{} score: {}".format("Classifier", score))
        print("Best params", clf.best_params_)
        print("Best estimator", clf.best_estimator_)

    def grid_search_classifier(self):

        classifier = LinearSVC()
        count_vect = TfidfVectorizer(stop_words = 'english')

        self.pipeline = Pipeline([
            ('vectorizer', count_vect),
            ('classifier', classifier)
        ])

        print(self.pipeline.get_params().keys())

        parameters = [{
            'classifier': (MultinomialNB(), LinearSVC(), LogisticRegression(), RandomForestClassifier(), MLPClassifier()),
            'vectorizer__binary': (True,),
            'vectorizer__lowercase': (True,),
            'vectorizer__max_df': (1.0,),
            'vectorizer__min_df': (2,),
            'vectorizer__ngram_range': ((1,2),),
            'vectorizer': (StemmedTfidfVectorizer(),)
        }]

        grid_search = GridSearchCV(self.pipeline, parameters, verbose = 3, n_jobs = -1)
        clf = grid_search.fit(self.trainsetX, self.trainsetY)
        score = clf.score(self.testsetX, self.testsetY)
        print("{} score: {}".format("NB", score))
        print("Best params", clf.best_params_)
        print("Best estimator", clf.best_estimator_)

def avg(lst):
    return sum(lst) / len(lst)

def stats(repeat=100):
    avgs = []

    avgs.append(avg([CarReviewsClassifier("data/car-reviews.csv").train_and_test() for i
                     in range(repeat)]))

    return avgs




if __name__ == "__main__":
    avgs = stats()
    print(avgs)
    #avgs.append(test_params(repeat=5, min_df=1, max_df=1, ngram_range=(1, 1), max_features=500))
    #avgs.append(test_params(repeat=5, min_df=1, max_df=1, ngram_range=(1, 1), max_features=1000))
    #avgs.append(test_params(repeat=1, min_df=1, max_df=0.9, ngram_range=(1, 1), max_features=7000))
    #print(avgs)
    #print(CarReviewsClassifier("data/car-reviews.csv").grid_search_vectorizer_params())




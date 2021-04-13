# Sentiment classifier car reviews

## Introduction
The goal of this work is to write a sentiment classifier.

### Comparing classifiers

````python
parameters = [{
            'classifier': (MultinomialNB(), LinearSVC(), LogisticRegression(), RandomForestClassifier(), MLPClassifier()),
            'vectorizer__binary': (True, False),
            'vectorizer__lowercase': (True,),
            'vectorizer__max_df': (1.0,),
            'vectorizer__min_df': (1,),
            'vectorizer__ngram_range': ((2,2),),
            'vectorizer': (TfidfVectorizer(),)
        }]
````

### Tuning hyper-parameters with GridSearch

A readily available tool for trying out different hyper-parameter values is the GridSearchCV class in the Scikit library. GridSearchCV implements a “fit” and a “score” method. It also implements “score_samples”, “predict”, “predict_proba”, “decision_function”, “transform” and “inverse_transform” if they are implemented in the estimator used. The parameters of the estimator used to apply these methods are optimized by cross-validated grid-search over a parameter grid. [2](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

The following hyper-parameters were tried out:

````python
'classifier__alpha': (0.01, 0.75, 1.0),
'vectorizer__binary': (True, False),
'vectorizer__lowercase': (True, False),
'vectorizer__max_df': (0.5, 0.9, 0.99),
'vectorizer__max_features': (1000, 5000, 10000),
'vectorizer__min_df': (1, 2, 3, 5),
'vectorizer__ngram_range': ((1,1), (2,2), (1,2), (1,3), (1,4), (3,3)),
'vectorizer': (CountVectorizer(), TfidfVectorizer(), StemmedCountVectorizer(), StemmedTfidfVectorizer())
````

````text
Loaded 1382 car reviews
dict_keys(['memory', 'steps', 'verbose', 'vectorizer', 'classifier', 'vectorizer__analyzer', 'vectorizer__binary', 'vectorizer__decode_error', 'vectorizer__dtype', 'vectorizer__encoding', 'vectorizer__input', 'vectorizer__lowercase', 'vectorizer__max_df', 'vectorizer__max_features', 'vectorizer__min_df', 'vectorizer__ngram_range', 'vectorizer__norm', 'vectorizer__preprocessor', 'vectorizer__smooth_idf', 'vectorizer__stop_words', 'vectorizer__strip_accents', 'vectorizer__sublinear_tf', 'vectorizer__token_pattern', 'vectorizer__tokenizer', 'vectorizer__use_idf', 'vectorizer__vocabulary', 'classifier__alpha', 'classifier__class_prior', 'classifier__fit_prior'])
Fitting 5 folds for each of 256 candidates, totalling 1280 fits
[CV 2/5] END classifier__alpha=0.01, vectorizer=CountVectorizer(), vectorizer__binary=True, vectorizer__lowercase=True, vectorizer__max_df=0.5, vectorizer__max_features=1000, vectorizer__min_df=1, vectorizer__ngram_range=(1, 1); total time=   0.5s
[CV 3/5] END classifier__alpha=0.01, vectorizer=CountVectorizer(), vectorizer__binary=True, vectorizer__lowercase=True, vectorizer__max_df=0.5, vectorizer__max_features=1000, vectorizer__min_df=1, vectorizer__ngram_range=(1, 1); total time=   0.5s
[CV 4/5] END classifier__alpha=0.01, vectorizer=CountVectorizer(), vectorizer__binary=True, vectorizer__lowercase=True, vectorizer__max_df=0.5, vectorizer__max_features=1000, vectorizer__min_df=1, vectorizer__ngram_range=(1, 1); total time=   0.5s
[CV 1/5] END classifier__alpha=0.01, vectorizer=CountVectorizer(), vectorizer__binary=True, vectorizer__lowercase=True, vectorizer__max_df=0.5, vectorizer__max_features=1000, vectorizer__min_df=1, vectorizer__ngram_range=(1, 1); total time=   0.5s
[CV 5/5] END classifier__alpha=0.01, vectorizer=CountVectorizer(), vectorizer__binary=True, vectorizer__lowercase=True, vectorizer__max_df=0.5, vectorizer__max_features=1000, vectorizer__min_df=1, vectorizer__ngram_range=(1, 1); total time=   0.6s
[CV 1/5] END classifier__alpha=0.01, vectorizer=CountVectorizer(), vectorizer__binary=True, vectorizer__lowercase=True, vectorizer__max_df=0.5, vectorizer__max_features=1000, vectorizer__min_df=3, vectorizer__ngram_range=(1, 1); total time=   0.5s
````
Choosing the classifier and best hyper-parameters based on the results of the grid search using 5-fold cross validation:

````text
NB score: 0.8686642599277978
Best params {'classifier': LinearSVC(), 'vectorizer': StemmedTfidfVectorizer(binary=True, min_df=2, ngram_range=(1, 2)), 'vectorizer__binary': True, 'vectorizer__lowercase': True, 'vectorizer__max_df': 1.0, 'vectorizer__min_df': 2, 'vectorizer__ngram_range': (1, 2)}
Best estimator Pipeline(steps=[('vectorizer',
                 StemmedTfidfVectorizer(binary=True, min_df=2,
                                        ngram_range=(1, 2))),
                ('classifier', LinearSVC())])
````



### Count Vectorizer

The initial implementation of the count vectorizer was implemented manually for learning purposes. The dataset is split into train and test dataset using pandas sample function, which selects desired number of rows randomly from a dataset:

````python
...
dataset = pd.read_csv(fpath)
self.trainset = dataset.sample(int(0.8 * dataset.shape[0]))
self.testset = dataset.drop(self.trainset.index)
...
````

The text is preprocessed in four steps:
* tokenization
* removal of stop words
* case conversion
* stemming

The initial versison of the count vectorizer was too slow. It needed approximately 1.5 seconds to vvectorize 100 reviews and over 52 seconds for the complete training dataset. Most of the time was spent on counting token occurences in order to generate the feature vectors:

```python
def tokenize(self, text):
    return nltk.tokenize.word_tokenize(text)

def remove_stop_words(self, word_list):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    filtered = [w for w in word_list if not w in stop_words]
    return filtered

def lower_case(self, word_list):
    return [w.lower() for w in word_list]

def stem_words(self, list_of_words):
    ps = nltk.stem.PorterStemmer()
    stemmed = [ps.stem(w) for w in list_of_words]
    return stemmed

def generate_features(self):
    features = []
    sentiments = []
    for r, sent in zip(self.trainset["Review"].to_list(), self.trainset["Sentiment"].to_list()):
        v = self.tokenize(r)
        c = self.remove_stop_words(v)
        l = self.lower_case(c)
        s = self.stem_words(l)
        features.append(s)
        sentiments.append(self.parse_sentiment_score(sent))

    self.prepare_bow(features)
    features = self.get_vectors(features)

    assert len(features) == len(sentiments)

    return features, sentiments
```
The generate_bow function was the culprit of slow execution:
```python
def generate_bow(list_tokenized_reviews):
    import numpy as np
    
    bow = {}
    for r in list_tokenized_reviews:
        for w in r:
            if w in bow:
                bow[w] += 1
            else:
                bow[w] = 1
    
    list_features = []
    for review in list_tokenized_reviews:
        vector = np.zeros(len(bow))
        for index, word in enumerate(bow):
            if word in review:
                vector[index] += 1
        list_features.append(vector)
    return list_features, bow
```

To make the vectorization faster the  repetitive vocabulary lookups when vectorizing an input had to be avoided. To overcome this the dictionary was replaced with an OrderedDict. Additionally an integeter index was added to each token in the vocabulary to facilitate the vectorization process:

```python
def generate_bow_faster(list_tokenized_reviews):
    # This is 6.5x faster than the previous version
    from collections import OrderedDict
    
    bow = {}
    for r in list_tokenized_reviews:
        for w in r:
            if w in bow:
                bow[w] += 1
            else:
                bow[w] = 1

    bow_ordered = OrderedDict(sorted(bow.items(), key=lambda t: t[0]))
    inx = 0
    for key in bow_ordered:
        bow_ordered[key] = (inx, bow_ordered[key])
        inx += 1
    list_features = []
    for review in list_tokenized_reviews:
        vector = [0] * len(bow)
        for i in range(len(review)):
            inx = bow_ordered[review[i]][0]
            vector[inx] += 1
        list_features.append(vector)
    return list_features, bow_ordered
``` 

The resulting ordered vocabulary where each element is a tuple containing a token and an index for faster lookup:

````text
('again', (1083, 47)), ('against', (1084, 2)), ('age', (1085, 51)), ('ageless', (1086, 1)), ('agenc', (1087, 7)), ('agent', (1088, 6)), ('aggrav', (1089, 6)), ('aggres', (1090, 3)), ('aggress', (1091, 27)), ('aggriv', (1092, 1)), ('agil', (1093, 7)), ('ago', (1094, 171)), ('agon', (1095, 1)), ('agoni', (1096, 1)), ('agre', (1097, 38)), ('agreeabl', (1098, 2)), ('agreement', (1099, 5)), ('agress', (1100, 5)), ('agressivli', (1101, 1)), ('ah', (1102, 6)), ('ahead', (1103, 41)), ('ahh', (1104, 1))
````
Using this method time to vectorize the dataset was brought down by more than 6.5x on avergage since we are avoiding repetitive word lookups. 

Since the scikit library offers all of this functionality out of the box with an additional benefit of being able to make use of the pipeline object the manual implementatin was replaced with the Scikit pipeline reducing the code size considerably. Instead of using pandas sample function to split dataset into train / test dataset this is now done with train_test_split function included in NLTK:

````python
...
dataset = pd.read_csv(fpath)

self.trainsetX, self.testsetX, self.trainsetY, self.testsetY = \
    train_test_split(dataset.Review, dataset.Sentiment, test_size=0.2)
...
````

To replicate the functionality of my simple count vectorizer the CountVectorizer class included in NLTK is used. The only addition needed to the default CountVectorizer  was stemming since this is not done by default. But instead of using the Porter stemmer an improved stemming algorithm called Snowball (Porter2) was used. With an slightly better execution time it offers better results. For stemming to be added in the pipeline the CountVectorizer class was overridden with a custom class:

````python
english_stemmer = nltk.stem.SnowballStemmer('english')
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([english_stemmer.stem(w) for w in analyzer(doc)])

````
so the basic NLTK pipeline looks like this:

```python
count_vect = StemmedCountVectorizer(analyzer="word", stop_words='english', binary=False)
classifier = MultinomialNB()
self.pipeline = Pipeline([
    ('vectorizer', count_vect),
    ('classifier', classifier)
])

self.pipeline.fit(self.trainsetX, self.trainsetY)
```
    

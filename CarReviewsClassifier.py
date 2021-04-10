import pandas as pd
import time
import matplotlib as plt
import nltk

class CarReviewsClassifier():

    def __init__(self):
        nltk.download('stopwords')
        nltk.download('punkt')

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


    # Function that generates bag of words and turns list of
    def generate_bow_faster(self, list_tokenized_reviews):
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

    def generate_features(self, df):
        t1 = time.time()
        features = []
        for r in df["Review"].to_list():
            v = self.tokenize(r)
            c = self.remove_stop_words(v)
            l = self.lower_case(c)
            s = self.stem_words(l)
            features.append(s)

        features = self.generate_bow_faster(features)
        t2 = time.time()
        print("Done in {:0.2f}sec".format(t2 - t1))
        #print(features)
        return features

    def train(self):
        dataset = pd.read_csv("data/car-reviews.csv")
        print("Loaded {} car reviews".format(dataset.shape[0]))

        trainset = dataset.sample(0.8*dataset.shape[0])
        print("Chosing random {} reviews for training".format(trainset.shape[0]))

        testset = dataset.drop(trainset.index)
        print("Testing on remaining {} reviews".format(testset.shape[0]))

        features = self.generate_features(trainset)
        print(features)


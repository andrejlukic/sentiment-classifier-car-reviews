from CarReviewsClassifier import *

def test_generate_features():
    classifier = CarReviewsClassifier("data/car-reviews.csv")
    classifier.train()
    classifier.test()
    #print(features)
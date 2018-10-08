#!/usr/bin/python

import random
import collections
import math
import sys
from util import *

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    return collections.Counter(x.split())
    # END_YOUR_CODE

############################################################
# Problem 3b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    weights = {}  # feature => weight
    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    random.seed(42)
    features = { x : featureExtractor(x) for x, _ in (trainExamples + testExamples)}
    indexes = [i for i in range(len(trainExamples))]
    for _ in range(numIters):
        random.shuffle(indexes)
        for i in indexes:
            x,y = trainExamples[i]
            if dotProduct(weights, features[x]) * y < 1:
                increment(weights, eta * y, features[x])
        predictor = lambda x: math.copysign(1, dotProduct(weights, features[x]))        
        print("Training error rate: {}.".format(evaluatePredictor(trainExamples, predictor)))
        print("Test error rate: {}.".format(evaluatePredictor(testExamples, predictor)))
    # END_YOUR_CODE
    return weights

############################################################
# Problem 3c: generate test case

def generateDataset(numExamples, weights):
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)
    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a nonzero score under the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    def generateExample():
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        phi = collections.Counter([random.choice(weights.keys())
            for _ in range(random.randint(0, 40*len(weights)))])
        score = dotProduct(weights, phi)
        if score == 0: return generateExample()
        y = math.copysign(1, score)
        # END_YOUR_CODE
        return (phi, y)
    return [generateExample() for _ in range(numExamples)]

############################################################
# Problem 3e: character features

def extractCharacterFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x):
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        counts = collections.Counter()
        string = ''.join(x.split())
        for i in range(0, len(string) - n + 1):
            counts[string[i:i+n]] += 1
        return counts
        # END_YOUR_CODE
    return extract

def testExtractCharacterFeatures(n):
    trainExamples = readExamples('polarity.train')
    devExamples = readExamples('polarity.dev')
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(trainExamples, devExamples, featureExtractor, numIters=20, eta=0.01)
    trainError = evaluatePredictor(trainExamples, lambda(x) : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    devError = evaluatePredictor(devExamples, lambda(x) : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print "Official on Character Features: train error = %s, dev error = %s" % (trainError, devError)
    return (trainError, devError)

def test():
    errors = []
    for n in range(20):
        print("======================= n = {} =================".format(n))
        errors.append((n, testExtractCharacterFeatures(n)))
    minTrain = min(errors, lambda p: p[1][0])
    print("Minimum training error at n = {} of {}".format(minTrain[0], minTrain[1]))
    minTest = min(errors, lambda p: p[1][1])
    print("Minimum test error at n = {} of {}".format(minTest[0], minTest[1]))

# test()
def interactiveWordFeatures():
    trainExamples = readExamples('polarity.train')
    devExamples = readExamples('polarity.dev')
    featureExtractor = extractWordFeatures
    weights = learnPredictor(trainExamples, devExamples, featureExtractor, numIters=20, eta=0.01)
    interactivePrompt(featureExtractor, weights)
# interactiveWordFeatures()

def interactive5GramModel():
    trainExamples = readExamples('polarity.train')
    devExamples = readExamples('polarity.dev')
    featureExtractor = extractCharacterFeatures(5)
    weights = learnPredictor(trainExamples, devExamples, featureExtractor, numIters=20, eta=0.01)
    interactivePrompt(featureExtractor, weights)

# interactive5GramModel()
############################################################
# Problem 4: k-means
############################################################
def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 32 lines of code, but don't worry if you deviate from this)
    centroids = random.sample(examples, K)
    def scale(v, scale):
        return { key : scale * value for key, value in v.iteritems() }
    
    phixi2 = [dotProduct(example, example) for example in examples]
    assignments = None
    prevAssignments = None
    totalLoss = None
    for _ in range(maxIters):
        # Compute new assignments and loss.
        muk2 = [dotProduct(centroid, centroid) for centroid in centroids]
        totalLoss = 0.0
        assignments = []
        newCentroids = [ collections.defaultdict(float) for _ in centroids]
        nSamplesInCentroid = [0 for _ in centroids]
        for i, example in enumerate(examples):
            distances = [phixi2[i] - 2*dotProduct(example, centroid) + muk2[k] for k, centroid in enumerate(centroids)]
            sampleLoss = min(distances)
            totalLoss += sampleLoss
            assignment = distances.index(sampleLoss)
            assignments.append(assignment)
            increment(newCentroids[assignment], 1, example)
            nSamplesInCentroid[assignment] += 1

        # No changes in assignments means we're done!
        if prevAssignments is not None and sum([abs(x-y) for x,y in zip(prevAssignments, assignments)]):
            return centroids, assignments, totalLoss

        # Otherwise, recompute centroids.
        prevAssignments = assignments
        centroids = [scale(centroid, 1.0 / n) for n, centroid in zip(nSamplesInCentroid, newCentroids)]

    return centroids, assignments, totalLoss 
    # END_YOUR_CODE

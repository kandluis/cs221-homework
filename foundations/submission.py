import collections
import math

############################################################
# Problem 3a

def findAlphabeticallyLastWord(text):
    """
    Given a string |text|, return the word in |text| that comes last
    alphabetically (that is, the word that would appear last in a dictionary).
    A word is defined by a maximal sequence of characters without whitespaces.
    You might find max() and list comprehensions handy here.
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return max(text.split())
    # END_YOUR_CODE

############################################################
# Problem 3b

def euclideanDistance(loc1, loc2):
    """
    Return the Euclidean distance between two locations, where the locations
    are pairs of numbers (e.g., (3, 5)).
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return math.sqrt(sum([(c1 - c2)**2 for c1, c2 in zip(loc1, loc2)]))
    # END_YOUR_CODE

############################################################
# Problem 3c
def mutateSentences(sentence):
    """
    Given a sentence (sequence of words), return a list of all "similar"
    sentences.
    We define a sentence to be similar to the original sentence if
      - it as the same number of words, and
      - each pair of adjacent words in the new sentence also occurs in the original sentence
        (the words within each pair should appear in the same order in the output sentence
         as they did in the orignal sentence.)
    Notes:
      - The order of the sentences you output doesn't matter.
      - You must not output duplicates.
      - Your generated sentence can use a word in the original sentence more than
        once.
    Example:
      - Input: 'the cat and the mouse'
      - Output: ['and the cat and the', 'the cat and the mouse', 'the cat and the cat', 'cat and the cat and']
                (reordered versions of this list are allowed)
    """
    # BEGIN_YOUR_CODE (our solution is 20 lines of code, but don't worry if you deviate from this)
    # Generate a map to "next" words.
    words = sentence.split()
    wordMapping = collections.defaultdict(set)
    for i in range(len(words) - 1):
        wordMapping[words[i]].add(words[i + 1])

    def generateValidSentences(startWord, sentenceLength):
        '''
        Recursive helper function that generates all valid sentences (based on the wordMapping) which being with 'startWord'
        and contain sentenceLength words. The returned set contains only unique sentences.
        '''
        if sentenceLength == 1: return set([startWord])
        sentences = set()
        for word in wordMapping[startWord]:
            sentences |= {"{} {}".format(startWord, sentence)
                for sentence in generateValidSentences(word, sentenceLength - 1)}
        return sentences

    result = set()
    for word in words:
        result |= generateValidSentences(word, len(words))
    return result

    # END_YOUR_CODE

############################################################
# Problem 3d

def sparseVectorDotProduct(v1, v2):
    """
    Given two sparse vectors |v1| and |v2|, each represented as collections.defaultdict(float), return
    their dot product.
    You might find it useful to use sum() and a list comprehension.
    This function will be useful later for linear classifiers.
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    return sum([v1[key] * v2[key] for key in set(v1.keys() + v2.keys())])
    # END_YOUR_CODE

############################################################
# Problem 3e

def incrementSparseVector(v1, scale, v2):
    """
    Given two sparse vectors |v1| and |v2|, perform v1 += scale * v2.
    This function will be useful later for linear classifiers.
    """
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    for key in set(v1.keys() + v2.keys()):
        v1[key] += scale * v2[key]
    # END_YOUR_CODE

############################################################
# Problem 3f

def findSingletonWords(text):
    """
    Splits the string |text| by whitespace and returns the set of words that
    occur exactly once.
    You might find it useful to use collections.defaultdict(int).
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    return {key for key, value in collections.Counter(text.split()).iteritems() if value == 1}
    # END_YOUR_CODE

############################################################
# Problem 3g

def computeLongestPalindromeLength(text):
    """
    A palindrome is a string that is equal to its reverse (e.g., 'ana').
    Compute the length of the longest palindrome that can be obtained by deleting
    letters from |text|.
    For example: the longest palindrome in 'animal' is 'ama'.
    Your algorithm should run in O(len(text)^2) time.
    You should first define a recurrence before you start coding.
    """
    # BEGIN_YOUR_CODE (our solution is 19 lines of code, but don't worry if you deviate from this)
    cache = {}
    def computeLongestPalindromeLengthHelper(i, j):
        '''
        Helper method that returns the longest palindrom that can be obtained by deleting
        letters from text[i...j] (inclusive)
        '''
        if (i,j) in cache:
            return cache[(i,j)]
        if i > j: return 0
        if i == j: return 1
        cache[(i,j)] = max(
            computeLongestPalindromeLengthHelper(i+1, j),
            computeLongestPalindromeLengthHelper(i, j-1),
            2 + computeLongestPalindromeLengthHelper(i + 1, j - 1) if text[i] == text[j] else 0)
        return cache[(i, j)]

    return computeLongestPalindromeLengthHelper(0, len(text) - 1)
    # END_YOUR_CODE

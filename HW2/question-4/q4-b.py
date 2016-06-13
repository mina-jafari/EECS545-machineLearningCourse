import os
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import cPickle
from sklearn.naive_bayes import MultinomialNB

def getFeature():
    with open(os.path.join('spam_filter_train.txt'), 'r') as f:
        trainData = f.readlines()
    with open(os.path.join('spam_filter_test.txt'), 'r') as f:
        testData = f.readlines()
    data = trainData + testData
    trainNum, testNum = len(trainData), len(testData)
    del trainData
    del testData

    targets = []
    for i in range(trainNum):
        targets.append(0) if data[i].split('\t')[0] == "ham" else targets.append(1)
    for i in range(len(data)):
        data[i] = data[i].replace('\n', '').split('\t')[1]
    # lemmatize
    lemmatized = []
    wnl = WordNetLemmatizer()
    for line in data:
        lemmatized.append([wnl.lemmatize(word) for word in line.split(' ')])
    # remove stopwords
    stopwordRemoved = []
    sw = set(stopwords.words('english'))
    for line in lemmatized:
        stopwordRemoved.append(' '.join([x for x in line if x not in sw]))
    # tf feature
    vec = CountVectorizer()
    features = vec.fit_transform(stopwordRemoved)

#my addition
    multiNom = MultinomialNB(alpha=1.8, fit_prior=True)
    multiNom.fit(features[:trainNum], targets)
    testTargets =  multiNom.predict(features[trainNum:])
    f = open('testTargets.csv', 'w')
    f.write("id,output\n")
    for i in range(1, len(testTargets)+1):
        f.write(str(i))
        f.write(",")
        f.write(str(testTargets[i-1]))
        f.write("\n")
    f.close()
#my addition

'''
    with open('trainFeatures.pkl', 'wb') as f:
        cPickle.dump(features[:trainNum], f)
    with open('testFeatures.pkl', 'wb') as f:
        cPickle.dump(features[trainNum:], f)
'''

def main():
    getFeature()
'''
    with open('trainFeatures.pkl', 'rb') as f:
         trainFeatures = cPickle.load(f)
    with open('testFeatures.pkl', 'rb') as f:
         testFeatures = cPickle.load(f)
'''


if __name__ == '__main__':
    main()

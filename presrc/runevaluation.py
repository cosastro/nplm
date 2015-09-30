from processgooglenews import *
import scipy.stats
import numpy
import pandas
import re


def loadEmbeddings(embeddingsFilePath, binary=True):
    if not binary:
        return loadWordVectors(embeddingsFilePath)

    return None


def evaluateRubensteinGoodenough(embeddings, filePath, base=10):
    with open(filePath) as file:
        lines = file.readlines()

    wordPairs = []
    targetScores = []
    for line in lines:
        word1, word2, score = tuple(line.strip().split('\t'))
        score = float(score)

        wordPairs.append((word1, word2))
        targetScores.append(score)

    scores = []
    for word1, word2 in wordPairs:
        word1Embedding = embeddings[word1]
        word2Embedding = embeddings[word2]

        score = cosineSimilarity(word1Embedding, word2Embedding)
        scores.append(score)

    pearson, pearsonDeviation = scipy.stats.pearsonr(scores, targetScores)
    spearman, spearmanDeviation = scipy.stats.spearmanr(scores, targetScores)

    rubensteinGoodenoughMetric = numpy.mean([pearson, spearman]) * base
    rubensteinGoodenoughMetricStateOfTheArt = 0.915 * base

    return rubensteinGoodenoughMetric, rubensteinGoodenoughMetricStateOfTheArt


def evaluateWordSimilarity353(embeddings, filePath, base=10):
    data = pandas.read_csv(filePath)

    wordPairs = []
    targetScores = []
    for word1, word2, score in zip(data['Word1'], data['Word2'], data['Score']):
        if word1 in embeddings and word2 in embeddings:
            wordPairs.append((word1, word2))
            targetScores.append(score)

    scores = []
    for word1, word2 in wordPairs:
        word1Embedding = embeddings[word1]
        word2Embedding = embeddings[word2]

        score = cosineSimilarity(word1Embedding, word2Embedding)
        scores.append(score)

    pearson, pearsonDeviation = scipy.stats.pearsonr(scores, targetScores)
    spearman, spearmanDeviation = scipy.stats.spearmanr(scores, targetScores)

    wordSimilarity353Metric = numpy.mean([pearson, spearman]) * base
    wordSimilarity353MetricStateOfTheArt = 0.81 * base

    return wordSimilarity353Metric, wordSimilarity353MetricStateOfTheArt


def evaluateSimLex999(embeddings, filePath, base=10):
    data = pandas.read_csv(filePath, sep='\t')

    wordPairs = []
    targetScores = []
    scores = []

    for word1, word2, targetScore in zip(data['word1'], data['word2'], data['SimLex999']):
        if word1 in embeddings and word2 in embeddings:
            wordPairs.append((word1, word2))

            targetScores.append(targetScore)

            word1Embedding = embeddings[word1]
            word2Embedding = embeddings[word2]

            score = cosineSimilarity(word1Embedding, word2Embedding)
            scores.append(score)

    pearson, pearsonDeviation = scipy.stats.pearsonr(scores, targetScores)
    spearman, spearmanDeviation = scipy.stats.spearmanr(scores, targetScores)

    simLex999Metric = numpy.mean([pearson, spearman]) * base
    simLex999MetricStateOfTheArt = 0.642 * base

    return simLex999Metric, simLex999MetricStateOfTheArt


def evaluateSyntacticWordRelations(embeddings, filePath, base=10, maxWords=5):
    with open(filePath, 'r') as file:
        lines = file.readlines()
        words = [tuple(line.lower().split(' ')) for line in lines if not line.startswith(':')]
        words = [(word0.strip(), word1.strip(), word2.strip(), word3.strip()) for word0, word1, word2, word3 in words]

    scores = []
    for word0, word1, word2, word3 in words:
        if word0 not in embeddings or word1 not in embeddings or word2 not in embeddings or word3 not in embeddings:
            continue

        word0Embedding = embeddings[word0]
        word1Embedding = embeddings[word1]
        word2Embedding = embeddings[word2]
        word3Embedding = embeddings[word3]

        similarity01 = cosineSimilarity(word0Embedding, word1Embedding)
        similarity23 = cosineSimilarity(word2Embedding, word3Embedding)

        score = 1
        minSimilarityDelta = abs(similarity01 - similarity23)
        for wordN in embeddings.keys()[:maxWords]:
            similarity2N = cosineSimilarity(word2Embedding, embeddings[wordN])
            similarityDelta = abs(similarity01 - similarity2N)

            score = not (similarityDelta < minSimilarityDelta)
            if not score:
                break

        scores.append(score)

    syntacticWordRelationsMetric = base * sum(scores) / len(scores)

    return syntacticWordRelationsMetric, base


def evaluateSATQuestions(embeddings, filePath, base=10):
    maxLineLength = 50
    aCode = ord('a')

    scores = []
    with open(filePath) as file:
        line = file.readline()
        while line != '':
            if len(line) < maxLineLength:
                match = re.match('(?P<word0>[\w-]+)\s(?P<word1>[\w-]+)\s[nvar]\:[nvar]', line)
                if match:
                    stemWord0, stemWord1 = match.group('word0'), match.group('word1')
                    validSample = stemWord0 in embeddings and stemWord1 in embeddings

                    if validSample:
                        stemWord0Embedding, stemWord10Embedding = embeddings[stemWord0], embeddings[stemWord1]
                        stemSimilarity = cosineSimilarity(stemWord0Embedding, stemWord10Embedding)

                    choiceSimilarityDeltas = []
                    line = file.readline()
                    match = re.match('(?P<word0>[\w-]+)\s(?P<word1>[\w-]+)\s[nvar]\:[nvar]', line)
                    while match:
                        choiceWord0, choiceWord1 = match.group('word0'), match.group('word1')
                        validSample = validSample and choiceWord0 in embeddings and choiceWord1 in embeddings

                        if validSample:
                            choiceWord0Embedding, choiceWord1Embedding = embeddings[choiceWord0], embeddings[choiceWord1]
                            choiceSimilarity = cosineSimilarity(choiceWord0Embedding, choiceWord1Embedding)

                            choiceSimilarityDelta = abs(stemSimilarity - choiceSimilarity)
                            choiceSimilarityDeltas.append(choiceSimilarityDelta)

                        line = file.readline()
                        match = re.match('(?P<word0>[\w-]+)\s(?P<word1>[\w-]+)\s[nvar]\:[nvar]', line)

                    if validSample:
                        choice = numpy.argmin(choiceSimilarityDeltas)
                        correctChoiceIndex = ord(line.strip()) - aCode
                        scores.append(int(choice == correctChoiceIndex))

            line = file.readline()

    satQuestionsMetric = base * sum(scores) / len(scores)
    satQuestionsMetricStateOfTheArt = 0.815 * base

    return satQuestionsMetric, satQuestionsMetricStateOfTheArt


if __name__ == '__main__':
    embeddingsFilePath = '../data/Text8-vectors/vectors.bin'

    embeddings = loadEmbeddings(embeddingsFilePath, False)

    rgFilePath = '../data/RG/EN-RG-65.txt'
    rgMetric, rgStateOfTheArt = evaluateRubensteinGoodenough(embeddings, rgFilePath)
    log.info('Rubenstein-Goodenough: {0:.2f}/10. State of the art: {1:.2f}/10'.format(rgMetric, rgStateOfTheArt))

    wordSimilarity353FilePath = '../data/WordSimilarity-353/combined.csv'
    wordSim353Metric, wordSim353StateOfTheArt = evaluateWordSimilarity353(embeddings, wordSimilarity353FilePath)
    log.info('WordSimilarity-353: {0:.2f}/10. State of the art: {1:.2f}/10'.format(wordSim353Metric, wordSim353StateOfTheArt))

    simLex999FilePath = '../data/SimLex-999/SimLex-999.txt'
    simLex999Metric, simLex999MetricStateOfTheArt = evaluateSimLex999(embeddings, simLex999FilePath)
    log.info('SimLex-999: {0:.2f}/10. State of the art: {1:.2f}/10'.format(simLex999Metric, simLex999MetricStateOfTheArt))

    syntWordRelFilePath = '../data/Syntactic-Word-Relations/questions-words.txt'
    syntWordRelMetric, syntWordRelMetricStateOfTheArt = evaluateSyntacticWordRelations(embeddings, syntWordRelFilePath)
    log.info('Syntactic word relations: {0:.2f}/10. State of the art: {1:.2f}/10'.format(syntWordRelMetric, syntWordRelMetricStateOfTheArt))

    satQuestionsFilePath = '../data/SAT-Questions/SAT-package-V3.txt'
    satQuestionsMetric, satQuestionsMetricStateOfTheArt = evaluateSATQuestions(embeddings, satQuestionsFilePath)
    log.info('SAT Questions: {0:.2f}/10. State of the art: {1:.2f}/10'.format(satQuestionsMetric, satQuestionsMetricStateOfTheArt))
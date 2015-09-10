import gzip
import cPickle
import log

class Kit():
    @staticmethod
    def loadData(dataFilePath):
        log.info('Loading data...', False)

        with gzip.open(dataFilePath) as file:
            trainingData, validationData, testData = cPickle.load(file)

        trainingX, trainingY = trainingData
        validationX, validationY = validationData
        testX, testY = testData

        message = 'Loading data... Done. Train({0}), validate({1}), test({2}).'.format(
            len(trainingX), len(validationX), len(testX))

        log.info(message, True)

        return trainingData, validationData, testData

    @staticmethod
    def test(testData):
        pass

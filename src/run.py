import kit
import nplm
import optimization

trainingData, validationData = kit.loadData()

model = nplm.Model()
trainingAlgorithm = optimization.TrainingAlgorithm()

trainingAlgorithm.train(model, trainingData, validationData)
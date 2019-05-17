package org.dl4j.sample

import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.io.File
import java.lang.IllegalStateException


fun main(args: Array<String>) {
    val dataDirectory = args.firstOrNull() ?: throw IllegalStateException("Data directory is not provided")
    train(dataDirectory)
}

private fun train(dataDirectoryString: String) {
    val rngSeed = 1337L
    val inputsCount = 4
    val labelsCount = 4
    val batchSize = 100

    @Suppress("SpellCheckingInspection") val dataDirectory = File(dataDirectoryString)

    val dataSets = mutableListOf<DataSet>()
    for (file in dataDirectory.listFiles()) {
        val recordReader = CSVRecordReader(',')
        recordReader.initialize(FileSplit(file))
        val iterator = RecordReaderDataSetIterator(recordReader, batchSize, 4, 7, true)
        dataSets.add(iterator.next())
    }

    val dataSet = DataSet.merge(dataSets)

    dataSet.shuffle(rngSeed)
    val testAndTrain = dataSet.splitTestAndTrain(0.8)
    val trainingData = testAndTrain.train
    val testData = testAndTrain.test

    @Suppress("UNUSED_VARIABLE") val testDataSetSize = testData.features.rows()
    @Suppress("UNUSED_VARIABLE") val trainDataSetSize = trainingData.features.rows()

    val conf = NeuralNetConfiguration.Builder()
        .seed(rngSeed) //include a random seed for reproducibility
        // use stochastic gradient descent as an optimization algorithm
        .updater(Nesterovs(0.006, 0.9))
        .l2(1e-4)
        .list()
        .layer(
            DenseLayer.Builder() //create the first, input layer with xavier initialization
                .nIn(inputsCount)
                .nOut(100)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .build()
        )
        .layer(
            OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
                .nIn(100)
                .nOut(labelsCount)
                .activation(Activation.SOFTMAX)
                .weightInit(WeightInit.XAVIER)
                .build()
        )
        .build()

    val model = MultiLayerNetwork(conf)
    model.init()
    //print the score with every 1 iteration
    model.setListeners(ScoreIterationListener(1))

    model.fit(trainingData)

    val iterator = DummyDataSetIterator(testData, labelsCount, inputsCount, batchSize)
    val evaluation = model.evaluate<Evaluation>(iterator)
    println(evaluation.stats())
}

package org.dl4j.sample

import org.deeplearning4j.datasets.iterator.DummyPreProcessor
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator

class DummyDataSetIterator(
    private val testData: DataSet,
    private val labelsCount: Int,
    private val featuresCount: Int,
    private val batchSize: Int) : DataSetIterator {

    private var currentIndex : Int = 0
    private var dataSetSize = testData.features.rows()

    private var preProcessor : DataSetPreProcessor = DummyPreProcessor()

    override fun resetSupported(): Boolean = true

    override fun getLabels(): MutableList<String>? = null

    override fun remove() {
    }

    override fun inputColumns(): Int = featuresCount

    override fun batch(): Int = batchSize

    override fun next(num: Int): DataSet {
        currentIndex = testData.features.rows()
        return testData
    }

    override fun next(): DataSet {
        currentIndex = testData.features.rows()
        return testData
    }

    override fun totalOutcomes(): Int = labelsCount

    override fun setPreProcessor(preProcessor: DataSetPreProcessor) {
        this.preProcessor = preProcessor
    }

    override fun reset() {
        currentIndex = 0
    }

    override fun hasNext(): Boolean = currentIndex < dataSetSize

    override fun asyncSupported(): Boolean = false

    override fun getPreProcessor(): DataSetPreProcessor = preProcessor

}
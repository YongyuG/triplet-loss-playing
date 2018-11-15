#!/usr/bin/env python
# -*- coding:utf8 -*-

import os
# force use cpu
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
from keras.layers import Input
from keras.layers import Reshape
from keras.models import Model
from keras.callbacks import LambdaCallback
from keras.layers import concatenate
import keras.backend as K
from keras.layers import Lambda
import numpy as np
import random
from keras.models import load_model


class TrainData(object):
    def __init__(self, featsFile, batchSize=100):
        self._speakerList = []
        self._speakerFeatsMap = {}
        self._loadFeats(featsFile)
        self._batchSize = batchSize

    def _loadFeats(self, featsFile):
        lastSpeaker = ""
        tmpM = []
        with open(featsFile) as f:
            for line in f:
                line = line.strip()
                if not line:
                    break
                if " " in line:
                    vec = []
                    for v in map(lambda x: float(x), line.split(" ")):
                        vec.append(v)
                    tmpM.append(vec)
                else:
                    if len(tmpM) > 0:
                        if lastSpeaker not in self._speakerFeatsMap:
                            self._speakerFeatsMap[lastSpeaker] = []
                        featsList = self._speakerFeatsMap[lastSpeaker]
                        featsList.append(np.array(tmpM))
                        self._speakerFeatsMap[lastSpeaker] = featsList
                    lastSpeaker = line
                    tmpM = []
        if len(tmpM) > 0:
            if lastSpeaker not in self._speakerFeatsMap:
                self._speakerFeatsMap[lastSpeaker] = []
                featsList = self._speakerFeatsMap[lastSpeaker]
                featsList.append(np.array(tmpM))
                self._speakerFeatsMap[lastSpeaker] = featsList
        for speaker in self._speakerFeatsMap.keys():
            self._speakerList.append(speaker)

    def getBatchCount(self):
        return len(self._speakerList) * 2

    def _getNegativeSample(self, excludeSpeaker):
        speakerCount = len(self._speakerList)
        while True:
            n = random.randint(0, speakerCount-1)
            if self._speakerList[n] == excludeSpeaker:
                continue
            sampleList = self._speakerFeatsMap[self._speakerList[n]]
            m = random.randint(0, len(sampleList)-1)
            # print(self._speakerList[n], m)
            # print("=============")
            return sampleList[m]

    def _randomSpeaker(self):
        n = random.randint(0, len(self._speakerList)-1)
        speaker = self._speakerList[n]
        featsList = self._speakerFeatsMap[speaker]
        m1 = random.randint(0, len(featsList)-1)
        m2 = random.randint(0, len(featsList)-1)
        # print(speaker, m1, m2)
        return featsList[m1], featsList[m2], speaker

    def iterSample(self):
        anchorList = []
        positiveList = []
        negativeList = []
        targets = []
        while True:
            anchor, positive, speaker = self._randomSpeaker()
            negative = self._getNegativeSample(speaker)
            if len(targets) >= self._batchSize:
                npAnchorList = np.array(anchorList)
                d1, d2, d3 = npAnchorList.shape
                npAnchorList = npAnchorList.reshape((d1, d2, d3, 1))
                npPositiveList = np.array(positiveList)
                npPositiveList = npPositiveList.reshape((d1, d2, d3, 1))
                npNegativeList = np.array(negativeList)
                npNegativeList = npNegativeList.reshape((d1, d2, d3, 1))
                yield [npAnchorList, npPositiveList, npNegativeList], np.array(targets)
                anchorList = []
                positiveList = []
                negativeList = []
                targets = []
            anchorList.append(anchor)
            positiveList.append(positive)
            negativeList.append(negative)
            targets.append(0.0)


def _cosine(a, b):
    return K.batch_dot(a, b, axes=1)


def _tripLoss(y_true, y_pred):
    p = _cosine(y_pred[:, 0], y_pred[:, 1])
    n = _cosine(y_pred[:, 0], y_pred[:, 2])
    loss = K.maximum(n-p+1.0, 0.0)
    return K.sum(loss)


def _defineLossModel(embeddingModel, rows, cols):
    _, embeddingSize = embeddingModel.output.shape
    anchor = Input((rows, cols, 1))
    positive = Input((rows, cols, 1))
    negative = Input((rows, cols, 1))

    embeddedAnchor = embeddingModel(anchor)
    embeddedPositive = embeddingModel(positive)
    embeddedNegative = embeddingModel(negative)
    embedList = concatenate(
        [embeddedAnchor, embeddedPositive, embeddedNegative])
    embedList = Reshape((3, int(embeddingSize)))(embedList)
    model = Model(inputs=[anchor, positive, negative], outputs=embedList)
    model.compile(optimizer="adam", loss=_tripLoss)
    return model


if __name__ == "__main__":
    if 4 != len(sys.argv):
        print("Usage:{} [featsFile] [preTrainModel] [outModelPath]".format(
            sys.argv[0]))
        sys.exit(1)
    featsFile = sys.argv[1]
    preTrainModel = sys.argv[2]
    outModelPath = sys.argv[3]
    model = load_model(preTrainModel, compile=False)
    _, rows, cols, _ = model.input.shape
    lossModel = _defineLossModel(model, rows, cols)
    trainData = TrainData(featsFile, 32)
    cb = LambdaCallback(on_epoch_end=lambda epoch, _: model.save(
        "{}/model_{}.h5".format(outModelPath, epoch)))
    lossModel.fit_generator(trainData.iterSample(), epochs=20,
                            steps_per_epoch=trainData.getBatchCount(), callbacks=[cb])

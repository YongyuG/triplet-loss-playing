#!/usr/bin/env python
# -*- coding:utf8 -*-

import sys
import numpy as np
import random
from keras.layers import GRU
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import GlobalAveragePooling1D
from keras.layers import concatenate
from keras.layers import Reshape
from keras.layers import Lambda
from keras.layers import Dropout
from keras.models import Model
import keras.backend as K
from keras.callbacks import LambdaCallback


class TrainData(object):
    def __init__(self, featsFile, negativeFactor=10, batchSize=100):
        self._speakerList = []
        self._speakerFeatsMap = {}
        self._totalFeatsCount = 0
        self._loadFeats(featsFile)
        self._negativeFactor = negativeFactor
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
                        self._totalFeatsCount += 1
                    lastSpeaker = line
                    tmpM = []
        if len(tmpM) > 0:
            if lastSpeaker not in self._speakerFeatsMap:
                self._speakerFeatsMap[lastSpeaker] = []
                featsList = self._speakerFeatsMap[lastSpeaker]
                featsList.append(np.array(tmpM))
                self._speakerFeatsMap[lastSpeaker] = featsList
                self._totalFeatsCount += 1
        for speaker in self._speakerFeatsMap.keys():
            self._speakerList.append(speaker)

    def getBatchCount(self):
        speakerCount = len(self._speakerList)
        samplePerSpeaker = int(self._totalFeatsCount / speakerCount)
        sampleCount = speakerCount*samplePerSpeaker * \
            (samplePerSpeaker-1)*self._negativeFactor
        return int(sampleCount/self._batchSize)+1

    def _listNegativeSample(self, excludeSpeaker):
        remainCount = self._negativeFactor
        speakerCount = len(self._speakerList)
        while remainCount > 0:
            n = random.randint(0, speakerCount-1)
            if self._speakerList[n] == excludeSpeaker:
                continue
            sampleList = self._speakerFeatsMap[self._speakerList[n]]
            m = random.randint(0, len(sampleList)-1)
            yield sampleList[m]
            remainCount -= 1

    def iterSample(self):
        anchorList = []
        positiveList = []
        negativeList = []
        targets = []
        while True:
            for speaker in self._speakerList:
                speakerFeats = self._speakerFeatsMap[speaker]
                segCount = len(speakerFeats)
                for i in range(segCount):
                    for j in range(segCount):
                        if i != j:
                            # i for anchor,j for positive
                            for negativeFeats in self._listNegativeSample(speaker):
                                if len(targets) >= self._batchSize:
                                    yield [np.array(anchorList), np.array(positiveList), np.array(negativeList)], np.array(targets)
                                    anchorList = []
                                    positiveList = []
                                    negativeList = []
                                    targets = []
                                anchorList.append(speakerFeats[i])
                                positiveList.append(speakerFeats[j])
                                negativeList.append(negativeFeats)
                                # inputs.append(
                                #     [np.array(speakerFeats[i]), np.array(speakerFeats[j]), np.array(negativeFeats)])
                                targets.append(0.0)


def _defineModel(frameCount, featCount):
    inputs = Input(shape=(frameCount, featCount))
    forward = None
    backward = None
    for i, outDim in enumerate([32, 64, 16]):
        if i:
            forward = GRU(outDim, return_sequences=True, activation="tanh",
                          dropout=0.5, recurrent_dropout=0.5)(forward)
            backward = GRU(outDim, return_sequences=True, activation="tanh",
                           dropout=0.5, recurrent_dropout=0.5, go_backwards=True)(backward)
        else:
            forward = GRU(outDim, return_sequences=True, activation="tanh",
                          dropout=0.5, recurrent_dropout=0.5)(inputs)
            backward = GRU(outDim, return_sequences=True, activation="tanh",
                           dropout=0.5, recurrent_dropout=0.5, go_backwards=True)(inputs)
    forward = GlobalAveragePooling1D()(forward)
    backward = GlobalAveragePooling1D()(backward)
    x = concatenate([forward, backward], axis=1)
    x = Dense(64, activation="tanh")(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation="tanh")(x)
    x = Dropout(0.5)(x)
    return Model(inputs=inputs, outputs=x)


def _tripLoss(y_true, y_pred):
    p = K.sum(K.square(y_pred[:, 0] - y_pred[:, 1]), axis=-1, keepdims=True)
    n = K.sum(K.square(y_pred[:, 0] - y_pred[:, 2]), axis=-1, keepdims=True)
    loss = K.maximum(0.0, p + 0.5 - n)
    return K.sum(loss)


def _defineLossModel(embeddingModel, frameCount, featCount):
    anchor = Input((frameCount, featCount))
    positive = Input((frameCount, featCount))
    negative = Input((frameCount, featCount))

    embeddedAnchor = embeddingModel(anchor)
    embeddedPositive = embeddingModel(positive)
    embeddedNegative = embeddingModel(negative)
    embedList = concatenate(
        [embeddedAnchor, embeddedPositive, embeddedNegative])
    embedList = Reshape((3, 32))(embedList)
    model = Model(inputs=[anchor, positive, negative], outputs=embedList)
    model.compile(optimizer="adam", loss=_tripLoss)
    return model


def _parseInputShape(featsFile):
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
                    return len(tmpM), len(tmpM[0])
    return 100, 13  # default value


if __name__ == "__main__":
    if 3 != len(sys.argv):
        print("Usage:{} [featsFile] [outModulePath]".format(sys.argv[0]))
        sys.exit(1)
    featsFile = sys.argv[1]
    outModulePath = sys.argv[2]
    frameCount, featCount = _parseInputShape(featsFile)
    print("seg param", frameCount, featCount)
    model = _defineModel(frameCount, featCount)
    lossModel = _defineLossModel(model, frameCount, featCount)
    trainData = TrainData(featsFile, 5, 2000)
    cb = LambdaCallback(on_epoch_end=lambda epoch, _: model.save(
        "{}/model_{}.h5".format(outModulePath, epoch)))
    lossModel.fit_generator(trainData.iterSample(), epochs=20,
                            steps_per_epoch=trainData.getBatchCount(), callbacks=[cb])

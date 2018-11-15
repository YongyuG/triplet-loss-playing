#!/usr/bin/env python
# -*- coding:utf8 -*-

import sys
import os
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import AveragePooling2D, GlobalAveragePooling1D
from keras.layers import BatchNormalization
from keras.layers import Dense, Reshape
from keras.models import Model
from keras.callbacks import LambdaCallback
from keras.layers import concatenate
import keras.backend as K
from keras.layers import Lambda
from keras import regularizers
from keras.layers import add
from keras.utils import to_categorical
import numpy as np
import random
import pickle


class TrainData(object):
    def __init__(self, featsDir, speakerPerEpoch=128, batchSize=64):
        self._allSpeakerList = []
        self._curSpeakerList = []
        self._speakerFeatsMap = {}
        self._totalFeatsCount = 0
        self._allSpeakerIdMap = {}
        self._batchSize = batchSize
        self._featsDir = featsDir
        self._speakerPerEpoch = speakerPerEpoch
        self._zeroVec = []

        self._loadSpeakerList()
        self.reload()

    def _loadSpeakerList(self):
        i = 0
        for name in os.listdir(self._featsDir):
            self._allSpeakerList.append(name)
            self._allSpeakerIdMap[name] = i
            i += 1

    def reload(self):
        print("reload train data")
        self._speakerFeatsMap = {}
        self._curSpeakerList = []
        self._totalFeatsCount = 0
        while len(self._speakerFeatsMap) < self._speakerPerEpoch:
            speakerId = random.randint(0, len(self._allSpeakerList)-1)
            speakerName = self._allSpeakerList[speakerId]
            if speakerName in self._speakerFeatsMap:
                continue
            with open(featsDir+"/"+speakerName, "rb") as f:
                segList = pickle.load(f)
                self._speakerFeatsMap[speakerName] = segList
                self._totalFeatsCount += len(segList)
        for speakerName, _ in self._speakerFeatsMap.items():
            self._curSpeakerList.append(speakerName)

    def getShape(self):
        d1 = 100
        d2 = 64
        for _, segList in self._speakerFeatsMap.items():
            for seg in segList:
                d1 = len(seg)
                d2 = len(seg[0])
        for i in range(d2):
            self._zeroVec.append(0.0)

        return d1, d2

    def getBatchCount(self):
        batchCount = (self._totalFeatsCount/self._batchSize)+1
        return batchCount * 10

    def getSpeakerCount(self):
        return len(self._allSpeakerList)

    def _randomCutOff(self, feats):
        ret = []
        p = random.randint(100, len(feats))
        for i, vec in enumerate(feats):
            if i < p:
                ret.append(vec)
            else:
                ret.append(self._zeroVec)
        return ret

    def iterSample(self):
        x = []
        y = []
        while True:
            n = random.randint(0, len(self._curSpeakerList)-1)
            speaker = self._curSpeakerList[n]
            cate = to_categorical(
                self._allSpeakerIdMap[speaker], len(self._allSpeakerList))
            fetasList = self._speakerFeatsMap[speaker]
            m = random.randint(0, len(fetasList)-1)
            feat = self._randomCutOff(fetasList[m])
            x.append(feat)
            y.append(cate)
            if len(x) >= self._batchSize:
                x = np.array(x)
                d1, d2, d3 = x.shape
                x = x.reshape((d1, d2, d3, 1))
                yield x, np.array(y)
                x = []
                y = []


def _clipped_relu(inp):
    return Lambda(lambda y: K.minimum(K.maximum(y, 0.0), 20.0))(inp)


def _res_block(inp, filters):
    x = Conv2D(filters, kernel_size=3, strides=1, activation=None, padding='same',
               kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(l=0.0001))(inp)
    x = BatchNormalization()(x)
    x = _clipped_relu(x)

    x = Conv2D(filters, kernel_size=3, strides=1, activation=None, padding='same',
               kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(l=0.0001))(x)
    x = BatchNormalization()(x)
    x = add([x, inp])
    return _clipped_relu(x)


def _conv_and_res_net(inp, filters):
    x = Conv2D(filters, kernel_size=5, strides=2, padding='same',
               kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(l=0.0001))(inp)
    x = BatchNormalization()(x)
    x = _clipped_relu(x)
    for _ in range(3):
        x = _res_block(x, filters)
    return x


def _defineModel(rows, cols):
    inputs = Input(shape=(rows, cols, 1))
    x = _conv_and_res_net(inputs, 64)
    x = _conv_and_res_net(x, 128)
    x = _conv_and_res_net(x, 256)
    _, d1, d2, d3 = x.shape
    x = Reshape((int(d1), int(d2)*int(d3)))(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64)(x)
    x = Lambda(lambda y: K.l2_normalize(y, axis=1))(x)
    return Model(inputs=inputs, outputs=x)


def _defineCateModel(embeddingModel, speakerCount, rows, cols):
    inp = Input((rows, cols, 1))
    x = embeddingModel(inp)
    x = Dense(speakerCount, activation="softmax")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(optimizer="nadam", loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def _epoch_cb(model, outModulePath, epoch, trainData):
    model.save("{}/model_{}.h5".format(outModulePath, epoch))
    trainData.reload()


if __name__ == "__main__":
    if 3 != len(sys.argv):
        print("Usage:{} [featsDir] [outModulePath]".format(sys.argv[0]))
        sys.exit(1)
    featsDir = sys.argv[1]
    outModulePath = sys.argv[2]
    trainData = TrainData(featsDir, 64, 64)
    rows, cols = trainData.getShape()
    model = _defineModel(rows, cols)
    speakerCount = trainData.getSpeakerCount()
    cateModel = _defineCateModel(model, speakerCount, rows, cols)
    cb = LambdaCallback(on_epoch_end=lambda epoch, _: _epoch_cb(
        model, outModulePath, epoch, trainData))
    cateModel.fit_generator(trainData.iterSample(), epochs=100,
                            steps_per_epoch=trainData.getBatchCount(), callbacks=[cb])

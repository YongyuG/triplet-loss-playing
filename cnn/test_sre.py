#!/usr/bin/env python
# -*- coding:utf8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
from keras.models import load_model
import speakin_voice_feats
import random
import numpy as np
from keras.models import Model
import math


_fbankExtractor = speakin_voice_feats.genFbankExtractor(
    allow_downsample=True, sample_frequency=8000, frame_length=25, frame_shift=10, high_freq=3700, low_freq=20, num_mel_bins=63)
_vadComputer = speakin_voice_feats.createVadComputer()
_threadHold = 0.5


def _iterSpeakerFiles(wavPath):
    for speaker in os.listdir(wavPath):
        wavFiles = []
        speakerPath = os.path.sep.join([wavPath, speaker])
        for rootPath, _, fileNames in os.walk(speakerPath):
            for fileName in fileNames:
                if not fileName.endswith(".wav"):
                    continue
                realPath = os.path.sep.join([rootPath, fileName])
                wavFiles.append(realPath)
        yield speaker, wavFiles


def _extractFeats(wavFile):
    fbankFeats = speakin_voice_feats.extractFeats(_fbankExtractor, wavFile)
    vadResult = speakin_voice_feats.computeVad(_vadComputer, fbankFeats)
    return fbankFeats, vadResult


def _genVec(model, mfccFeats, vadResult, frameCount, featsCount):
    frameSize = len(mfccFeats)
    if frameSize < frameCount+50:
        return None
    remainCount = 10
    predictFeatsList = []
    while remainCount > 0:
        n = random.randint(0, frameSize-frameCount-1)
        vad = vadResult[n:n+frameCount]
        totalVad = 0.0
        for v in vad:
            totalVad += v
        if totalVad > frameCount * 0.8:
            predictFeatsList.append(mfccFeats[n:n+frameCount])
            remainCount -= 1
    npPredictFeatsList = np.array(predictFeatsList)
    d1, d2, d3 = npPredictFeatsList.shape
    npPredictFeatsList = npPredictFeatsList.reshape((d1, d2, d3, 1))
    vecList = model.predict(npPredictFeatsList)
    return np.array(vecList).mean(axis=0)


def _extractVec(model, wavFiles, frameCount, featsCount):
    vecList = []
    for wavFile in wavFiles:
        mfccFeats, vadResult = _extractFeats(wavFile)
        vec = _genVec(model, mfccFeats, vadResult, frameCount, featsCount)
        if vec is None:
            continue
        print(vec)
        vecList.append(vec)
    return vecList


def _genDistance(vec1, vec2):
    # cosine
    return np.sum(vec1*vec2)/np.sqrt(np.sum(vec1*vec1))/np.sqrt(np.sum(vec2*vec2))


def _cmpDistance(speakerVecList):
    sameList = []
    diffList = []
    rightCount = 0
    errCount = 0
    sameRealCount = 0
    sameErrCount = 0
    for speaker1, vec1 in speakerVecList:
        for speaker2, vec2 in speakerVecList:
            dis = _genDistance(vec1, vec2)
            if speaker1 == speaker2:
                sameRealCount += 1
                sameList.append(dis)
                if dis >= _threadHold:
                    rightCount += 1
                else:
                    errCount += 1
                    sameErrCount += 1
            else:
                diffList.append(dis)
                if dis < _threadHold:
                    rightCount += 1
                else:
                    errCount += 1
    sameList = np.array(sameList)
    diffList = np.array(diffList)
    print("same", np.percentile(sameList, 5),
          np.percentile(sameList, 50), np.percentile(sameList, 95))
    print("diff", np.percentile(diffList, 5),
          np.percentile(diffList, 50), np.percentile(diffList, 95))
    print("rightCount", rightCount)
    print("errCount", errCount)
    print("sameRealCount", sameRealCount)
    print("sameErrCount", sameErrCount)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:{} [modelPath] [wavPath]".format(sys.argv[0]))
        sys.exit(1)
    modelPath = sys.argv[1]
    wavPath = sys.argv[2]
    model = load_model(modelPath, compile=False)    
    model.compile(optimizer="adam", loss="mean_absolute_error")
    _, frameCount, featsCount, _ = model.inputs[0].shape
    model._make_predict_function()
    speakerVecList = []
    speakerSet = set()
    for speaker, wavFiles in _iterSpeakerFiles(wavPath):
        if len(wavFiles) < 3:
            continue
        wavFiles = wavFiles[:3]
        vecList = _extractVec(model, wavFiles, int(
            frameCount), int(featsCount))
        for vec in vecList:
            speakerVecList.append((speaker, vec))
        speakerSet.add(speaker)
        if len(speakerSet) >= 50:
            break
    _cmpDistance(speakerVecList)

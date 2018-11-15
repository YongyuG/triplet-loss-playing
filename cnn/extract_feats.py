#!/usr/bin/env python
# -*- coding:utf8 -*-


import argparse
import os
import speakin_voice_feats
import random
import pickle


def _parse_cmd_line():
    parser = argparse.ArgumentParser(description="extract feats from wav")
    parser.add_argument("--segPerSpeaker", type=int,
                        action="store", default=40)
    parser.add_argument("--framePerSeg", type=int, action="store", default=100)
    parser.add_argument("wavDir", type=str, action="store")
    parser.add_argument("featsOutDir", type=str, action="store")
    return parser.parse_args()


def _iter_speaker_wav(baseDir):
    for speakerName in os.listdir(baseDir):
        speakerDir = os.path.sep.join([baseDir, speakerName])
        wavFileList = []
        for wavFile in os.listdir(speakerDir):
            if wavFile.endswith(".wav"):
                wavFilePath = os.path.sep.join([speakerDir, wavFile])
                wavFileList.append(wavFilePath)
        yield speakerName, wavFileList


def _extract_feats(wavFiles):
    fbankExtractor = speakin_voice_feats.genFbankExtractor(
        allow_downsample=True, sample_frequency=8000, frame_length=25, frame_shift=10, high_freq=3700, low_freq=20, num_mel_bins=63)
    vadComputer = speakin_voice_feats.createVadComputer()
    for wavFile in wavFiles:
        fbankFeats = speakin_voice_feats.extractFeats(fbankExtractor, wavFile)
        vadResult = speakin_voice_feats.computeVad(vadComputer, fbankFeats)
        yield fbankFeats, vadResult


def _extract_seg(wavFiles, segPerSpeaker, framePerSeg):
    featsList = []
    for fbankFeats, vadResult in _extract_feats(wavFiles):
        featsList.append([fbankFeats, vadResult, len(vadResult)])
    segList = []
    iterCount = 0
    while len(segList) < segPerSpeaker and iterCount < segPerSpeaker*10:
        iterCount += 1
        n = random.randint(0, len(featsList)-1)
        featsLen = featsList[n][2]
        if featsLen <= framePerSeg:
            continue
        m = random.randint(0, featsLen-framePerSeg)
        seg = featsList[n][0][m:m+framePerSeg]
        segVad = featsList[n][1][m:m+framePerSeg]
        totalVad = 0.0
        for v in segVad:
            totalVad += v
        if totalVad >= 0.8*framePerSeg:
            segList.append(seg)
    return segList


if __name__ == "__main__":
    args = _parse_cmd_line()
    os.makedirs(args.featsOutDir, exist_ok=True)
    for speakerName, wavFiles in _iter_speaker_wav(args.wavDir):
        segList = _extract_seg(wavFiles, args.segPerSpeaker, args.framePerSeg)
        if len(segList) != args.segPerSpeaker:
            continue
        with open(args.featsOutDir+"/"+speakerName, "wb") as f:
            pickle.dump(segList, f)
            f.flush()
        print("done", speakerName)

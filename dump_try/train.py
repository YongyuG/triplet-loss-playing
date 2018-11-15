#-*- coding:utf-8 -*-
import sys
import numpy as np
import random



class TrainData(object):
    def __init__(self,featsFile, negativeFactor = 10, batchSize=100):
        self._speakerList = []
        self._speakerFeatsMap = {}
        self._totalFeatsCount = 0
        self._loadFeats(featsFile)
        self._negativeFactor = negativeFactor
        self._batchSize = batchSize

    def _loadFeats(self,featsFile): # 数据处理部分,用map存人和他的音频
        lastSpeaker = ""
        tmpM=[]
        with open(featsFile) as f:
            for line in f:
                line = line.strip()
                if not line:
                    break
                if " " in line:
                    vec=[]
                    for v in map(lambda x:float(x),line.split(" ")):
                        vec.append(v)   # 这里是一帧的MFCC
                    tmpM.append(vec)    # 把每一个人的mfcc都装进来,因为文件有人的名字,所以不加入人的名字
                                        # 这里就是一个音频的mFCC 上面是对数据类型做一些简单的转换, 一个音频我们假设有100帧数据
                else:
                    print(line)         # 当遇到名字的时候
                    if len(tmpM) >0:
                        if lastSpeaker not in self._speakerFeatsMap:
                            self._speakerFeatsMap[lastSpeaker]=[]
                        featsList=self._speakerFeatsMap[lastSpeaker]
                        featsList.append(np.array(tmpM))   # 把同一个人的 不通音频的MFCC 加载到字典里面来,即MAP
                        self._speakerFeatsMap[lastSpeaker] = featsList
                        self._totalFeatsCount += 1         # 这里是数总共有多少个音频mfcc,就是不考虑不同人
                    lastSpeaker = line   # 上面的操作都是对鹏到line时上一个说话人的操作,所以要把上一个说话人复制到lastspeaker上
                    tmpM=[]  # 遇到不同的音频时,的进行重置操作

        if len(tmpM) > 0:    # 这是进行最后一个人的情况
            if lastSpeaker not in self._speakerFeatsMap:
                self._speakerFeatsMap[lastSpeaker] = []
            featsList = self._speakerFeatsMap[lastSpeaker]
            featsList.append(np.array(tmpM))
            self._speakerFeatsMap[lastSpeaker] = featsList
            self._totalFeatsCount += 1
            print('last ',self._totalFeatsCount)

        for speaker in self._speakerFeatsMap.keys():  # 得到不同说话人的ID或者说名字
            self._speakerList.append(speaker)

    def getBatchCount(self):
        speakerCount = len(self._speakerList)

        samplePerSpeaker = int(self._totalFeatsCount / speakerCount) #每个人有40个样本,即一个说话人有40个音频
        sampleCount = speakerCount * samplePerSpeaker * \
                      (samplePerSpeaker - 1) * self._negativeFactor
        return int(sampleCount / self._batchSize)+1  #每个batch多少个样本

    def _listNegativeSample(self,excludeSpeaker):
        remainCount = self._negativeFactor
        speakerCount=len(self._speakerList)
        while remainCount > 0:
            n = random.randint(0,speakerCount-1)
            if self._speakerList[n] == excludeSpeaker:
                continue
            sampleList = self._speakerFeatsMap[self._speakerList[n]]
            m = random.randint(0,len(sampleList)-1)
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
                        if i !=j:       #i for anchor, j for positive
                            for negativeFeats in self._listNegativeSample(speaker):
                                if len(targets) >= self._batchSize:
                                    yield [np.array(anchorList), np.array(positiveList), np.array(negativeList), np.array(targets)]
                                    anchorList = []
                                    positiveList = []
                                    negativeList = []
                                    targets =[]
                                anchorList.append(speakerFeats[i])
                                positiveList.append(speakerFeats[i])
                                negativeList.append(negativeFeats)
                                targets.append(0.0)




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

if __name__ == '__main__':
    if len(sys.argv)!=2:
        print('Usage: {} [featsFile] [outModelPath]'.format(sys.argv[0]))
        sys.exit(1)
    featsFile = sys.argv[1]
    #outModelPath = sys.argv[2]
    framCount,featCount = _parseInputShape(featsFile)
    print('seg param ',framCount,featCount)
    trainData = TrainData(featsFile, 5, 2000)

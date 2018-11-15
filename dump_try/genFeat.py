#-*- coding:utf-8 -*-
import argparse
import os
import speakin_voice_feats
import random
import numpy as np



def _parse_cmd_line():
    parser=argparse.ArgumentParser(description="extract feats from wav")
    parser.add_argument("--segPerSpeaker",type=int,
                        action='store',default=40)
    parser.add_argument("--framePerSeg",type=int,action='store',default=100)
    parser.add_argument("wavDir",type=str,action='store')
    parser.add_argument('featsOutFile',type=str,action='store')
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

def _extract_feats(wavFiles):  ###  提取特征
    mfccExtracotr=speakin_voice_feats.genMfccExtractor(
        allow_downsample=True, sample_frequency=8000,frame_length=25,frame_shift=10,high_freq=3700,low_freq=20)
    vadComputer=speakin_voice_feats.createVadComputer()
    for wavFile in wavFiles:
        mfccFeats=speakin_voice_feats.extractFeats(mfccExtracotr,wavFile)
        vadresult=speakin_voice_feats.computeVad(vadComputer,mfccFeats)
        yield mfccFeats,vadresult

def _extract_seg(wavfile,segPerSpk=40,framePerSeg=100):  #把好几帧的片段拼接起来,形成输入,这里是100帧,对每个说话人,只抽取其40个音频
    featsList=[]
    for mfccFeats, vadResult in _extract_feats(wavfile):
        featsList.append([mfccFeats,vadResult,len(vadResult)])
    segList=[]
    iterCount=0

    while len(segList)<segPerSpk and iterCount<segPerSpk*10:
        iterCount+=1
        n=random.randint(0,len(featsList)-1) #随机抽样到 第n个样本
        featsLen=featsList[n][2] #拿到第n个样本的音频长度,即有多少帧
        if featsLen<=framePerSeg:  #如果该样本的特征长度小于我们规定的每个输入音频的长度
            continue #把音频给舍弃掉
        m = random.randint(0,featsLen-framePerSeg) #随机从取到该音频某个帧到段所需长度的音频帧
        seg = featsList[n][0][m:m+framePerSeg]  #截取从选取的索引开始到之后的特定长度的音频
        segVad = featsList[n][1][m:m+framePerSeg] #同上vad
        totalVad = 0.0


        for v in segVad:
            totalVad+=v
        if totalVad >-0.8*framePerSeg: #检测有多少有效语音,如果超过80都是有效语音
            segList.append(seg)
    return segList


if __name__ == '__main__':

    args=_parse_cmd_line() #获取输入
    outFile = open(args.featsOutFile,"w")
    for speakerName,wavFiles in _iter_speaker_wav(args.wavDir):
        segList=_extract_seg(wavFiles,args.segPerSpeaker,args.framePerSeg)  #存了每一个说话人的四十个音频的mfcc,每个MFCC一个有200帧,
        print(np.array(segList).shape)
        if len(segList) != args.segPerSpeaker:#如果一个人没有40个音频文件来做训练
            continue
        for seg in segList: #每个音频选择的mfcc
            outFile.write(speakerName)
            outFile.write('\n')
            for frame in seg:
                frame = map(lambda x:str(x),frame)
                line = " ".join(frame)
                outFile.write(line)
                outFile.write('\n')
        print('done {}'.format(speakerName))
    outFile.flush()
    outFile.close()


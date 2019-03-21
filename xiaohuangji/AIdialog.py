import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import thulac
import itertools

MAX_LENGTH = 20
MIN_COUNT = 5
PAD_token = 0   #padding 对应的索引
SOS_token = 1   #Start of Sentence 对应的索引
EOS_token = 2   #End of Sentence  对应的索引
small_batch_size = 5 
#想要加速训练或者想要利用GPU并行计算能力，
#则需要使用小批量（mini-batches）来训练。

class Voc:
    def __init__(self,name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num = 3  # Count SOS, EOS, PAD

    #Add a word to our vocabulary
    def add_word(self,word):
        if word not in self.word2index:
            self.word2index[word] = self.num
            self.word2count[word] = 1
            self.index2word[self.num] = word
            self.num += 1
        else:
            self.word2count[word] += 1

    #Add a sentence to our vocabulary
    def add_sentence(self,sentence):
        for word in sentence.split(' '):
            self.add_word(word)
    
    #Delete the words below a certain count threshold
    def trim(self,min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num = 3  # Count SOS, EOS, PAD

        for word in keep_words:
            self.add_word(word)

def read_voc(corpus_name,datafile):
    thu1 = thulac.thulac(seg_only = True)
    lines = open(datafile,encoding = 'utf-8').read().strip().split('\n')
    pairs = []
    for i in range(len(lines)):
        if lines[i]=='E':   #Start of conversation
            try:
                p1 = re.search(r'[^M]+',lines[i+1]).group().strip()
                p2 = re.search(r'[^M]+',lines[i+2]).group().strip()
            except:
                continue
            p1 = thu1.cut(p1,text = True)
            p2 = thu1.cut(p2,text = True)
            pairs.append([p1,p2])
    
    voc = Voc(corpus_name)
    return voc, pairs

# Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filterPair(p):
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

# Filter pairs using filterPair condition
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def load_data(corpus_name,datafile):
    print('Start preparing training data ...')
    voc, pairs = read_voc(corpus_name,datafile)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words ...")
    for pair in pairs:
        voc.add_sentence(pair[0])
        voc.add_sentence(pair[1])
    print("Counted words:", voc.num)
    return voc, pairs

def trim_data(voc,pairs,MIN_COUNT):
    voc.trim(MIN_COUNT)

    keep_pairs = []
    for pair in pairs:
        s1 = pair[0]
        s2 = pair[1]
        keep_s1 = True
        keep_s2 = True
        for word in s1.split(' '):
            if word not in voc.word2index:
                keep_s1 = False
                break
        for word in s2.split(' '):
            if word not in voc.word2index:
                keep_s2 = False
                break
        if keep_s1 and keep_s2:
            keep_pairs.append(pair)
    return keep_pairs

def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]

def zeroPadding(index_batch, fillvalue=PAD_token):
    return list(itertools.zip_longest(*index_batch, fillvalue=fillvalue))

#tensor : max_length * batch_size
def binaryMatrix(List,value = PAD_token):
    m = []
    for i,seq in enumerate(List):
        m.append([])
        for token in seq:
            if token ==PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m 



def batch2TrainData(voc,pair_batch):
    #按照对话中的第一句话的长度排序，降序
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    # input(对话的上一句)
    indexes_batch_in = [indexesFromSentence(voc, sentence) for sentence in input_batch]
    lenList_in = torch.tensor([len(index) for index in indexes_batch_in])
    padList_in = zeroPadding(indexes_batch_in)
    padTensor_in = torch.LongTensor(padList_in)
    
    #output(对话的下一句)
    indexes_batch_out = [indexesFromSentence(voc, sentence) for sentence in output_batch]
    max_target_len = max([len(index) for index in indexes_batch_out])
    padList_out = zeroPadding(indexes_batch_out)
    padTensor_out = torch.LongTensor(padList_out)
    #张量-->mask
    mask = torch.ByteTensor(binaryMatrix(padList_out))

    return padTensor_in,lenList_in,padTensor_out,mask,max_target_len





#batches = batch2TrainData(voc,[random.choice(pairs) for _ in range(small_batch_size)])
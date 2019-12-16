#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:juzphy
# datetime:2019-11-20 11:51

import numpy as np
import jieba as jb
import torch
import torch.nn.functional as F

"""
skip-gram, x为当前词ont-hot，y为当前词前后window个词的one-hot 
train data: 381*91, 91个不同词
        w1: 91*10
        w2: 10*91
   输出的w1: 91*10，即代表91个x词所对应的10维向量         
"""


def to_one_hot(data_point_index, vocab_size):
    """
    对单词进行one-hot representation
    :param data_point_index: 单词在词汇表的位置索引
    :param vocab_size: 词汇表大小
    :return: 1 x vocab_size 的one-hot representatio
    """
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp


def gen_data(string, window_size=2):
    stop_words = ['经', '，', '是', '、', '等', '和', '的', '。', '非', '虽然', '不是', '于', '但', '对', '到', '了','让', '人',
                  '过', '为', '都', '以', '也', '正是', '他们', '如今', '一种', '通过', '可以', '一个', '进行', '在', '更', '均',
                  '有', '取得', '’', '（', '）', '为了', '具有', ' ', '当前', '接近', '保证', '一组', '多个', '用于', '且', '就是',
                  '这种', '上', '它', '相比', '其中', '能', '尽可能', '及', '与', '为了', '加','可', '最为', '无', '性']
    words = [w for w in jb.cut(string) if w not in stop_words]
    words_set = set(words)
    words_size = len(words_set)
    word2int, int2word = {}, {}
    for index, word in enumerate(words_set):
        word2int[word] = index
        int2word[index] = word

    sentences = []
    for s in string.split('。'):
        sentences.append([st for st in jb.lcut(s) if st not in stop_words])

    x_train, y_train = [], []
    for sentence in sentences:
        for word_index, word in enumerate(sentence):
            for nb_word in sentence[max(word_index-window_size, 0): min(word_index+window_size, len(sentence)+1)]:
                if nb_word != word:
                    x_train.append(to_one_hot(word2int[word], words_size))
                    y_train.append(to_one_hot(word2int[nb_word], words_size))
    print(f'Total num of word is: {words_size}')
    return np.array(x_train), np.array(y_train, dtype='int64'), words_size, word2int, int2word


def train_embedding(x_train, y_train, vocab_size, embedding_dim=5, num_iterations=10000, learning_rate=0.1):
    print(f'Train data shape: {x_train.shape}, label y: {y_train.shape}')
    w1 = torch.randn(vocab_size, embedding_dim, dtype=float, requires_grad=True)
    b1 = torch.randn(embedding_dim, dtype=float, requires_grad=True)
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    hidden_representation = torch.add(torch.matmul(x_train, w1), b1)
    hidden_size = hidden_representation.shape
    print(f"hidden layer shape: ({hidden_size[0]}, {hidden_size[1]})")
    w2 = torch.randn(embedding_dim, vocab_size, dtype=float, requires_grad=True)
    b2 = torch.randn(vocab_size, dtype=float, requires_grad=True)
    for it in range(1, num_iterations+1):
        prediction = F.softmax(torch.add(torch.matmul(hidden_representation, w2), b2), dim=1)
        loss = -1 * torch.sum(y_train * torch.log(prediction), dim=1).mean()
        if it % 1000 == 0:
            print(f'iteration: {it}, loss: {loss.item()}')
        loss.backward(retain_graph=True)

        with torch.no_grad():
            w1.data -= learning_rate * w1.grad
            w2.data -= learning_rate * w2.grad
            b1.data -= learning_rate * b1.grad
            b2.data -= learning_rate * b2.grad
            w1.grad.data.zero_()
            w2.grad.data.zero_()
            b1.grad.data.zero_()
            b2.grad.data.zero_()
    vectors = (w1+b1).detach().numpy()
    print(f'vector shape:{vectors.shape}')
    return vectors


def calc_euclidean_distance(vec1, vec2):
    return np.sqrt(np.sum(np.power(vec1 - vec2, 2)))


def search_closest(word_index, vectors, word_dict, topk=10):
    query_vector = vectors[word_index]
    top_list = [(word_dict[index], calc_euclidean_distance(v, query_vector)) for index, v in enumerate(vectors) if index != word_index]
    top_list = sorted(top_list, key=lambda x: x[1])[:topk]
    return top_list


if __name__ == "__main__":
    documents = "MLP网络是一种应用最为广泛的一种网络，其中DNN就是属于MLP网络，它是一个前向结构的人工神经网络，输入一组向量向前传播输出向量。" \
                "RNN是一种节点定向连接成环的人工神经网络，与DNN网络相比，RNN可以利用上一个时序的输出及当前输入计算输出。" \
                "卷积神经网络，是一种前馈神经网络，通过卷积操作可以对一个连续区域进行识别，在图像处理取得不错效果。卷积神经网络的结构有" \
                "原始图像输入层、卷积层、池化层、全连接层、输出层。AE自编码器，属于无监督网络。自编码器的目的是输入X与输出X’尽可能接近，" \
                "网络结构为两层的MLP，这种接近程度通过重构误差表示，误差的函数有均方差和交叉熵，为了保证网络的稀疏性误差函数加L1正则项，" \
                "为了保证网络的鲁棒性输入增加随机噪声数据。Restricted Boltzmann Machine（受限波尔兹曼机 ）RBM是无监督的网络。" \
                "具有两层结构、对称连接且无自反馈的随机神经网络模型，层间全连接，层内无连接。RBM是一种有效的特征提取方法，用于初始化" \
                "前馈神经网络可明显提高泛化能力，堆叠多个RBM组成的深度信念网络（DBN）能提取更抽象的特征。"
    train_x, train_y, words_len, w2i, i2w = gen_data(documents, window_size=2)
    vector = train_embedding(train_x, train_y, words_len, embedding_dim=10)
    search_word = '神经网络'
    topk_closest = search_closest(w2i[search_word], vector, i2w, topk=10)
    for i in topk_closest:
        print(f'与{search_word}接近的词有：{i[0]}, 距离为：{i[1]}')

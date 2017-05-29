# -*- coding: utf-8 -*-
# @Time    : 2017/5/10 15:39
# @Author  : Studog
# @File    : model_train.py

import tensorflow as tf
from sklearn.cross_validation import train_test_split

max_name_length = 8

def dataproprecessing():
    name_dataset = r'.\person_1324473.txt'
    name = []
    gender = []
    name_vec = []

    with open(name_dataset, 'r', encoding='utf-8') as f:
        try:
            for line in f:
                content = line.strip().split(',')
                name.append(content[0])
                if content[1] == '男':
                    gender.append([0, 1])
                else:
                    gender.append([1, 0])
        except UnicodeDecodeError as e:
            print(e)
            pass

    vocabulary = {}
    for n in name:
        tokens = [word for word in n]
        for w in tokens:
            vocabulary[w] = vocabulary.setdefault(w, 0) + 1

    vocabulary_list = [' '] + sorted(vocabulary, key=vocabulary.get, reverse=True)
    vocab_size = len(vocabulary_list)
    vocab = dict([(x, y) for (y, x) in enumerate(vocabulary_list)])

# 把名字和性别对应起来
    for n in name:
        word_vec = []
        for word in n:
            word_vec.append(vocab.get(word))
        while len(word_vec) < max_name_length:
            word_vec.append(0)
        name_vec.append(word_vec)
    return name_vec, gender, vocab_size, vocab

# http://blog.csdn.net/u013713117/article/details/55049808
def neural_network(vocab_size, embedding_size=128, num_filters=128):
    # 将单词索引映射到低维的向量表示
    with tf.name_scope("embedding"):
        # W是在训练时得到的embedding matrix.，用随机均匀分布进行初始化
        W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
        # 实现embedding操作，得到一个3-dimensional的张量
        # 根据X的位置返回W在相同位置的参数
        embedded_chars = tf.nn.embedding_lookup(W, X)
        # conv2d 需要四个参数， 分别是batch, width, height 以及channel
        # embedding之后不包括 channel, 所以我们人为地添加上它，并设置为1
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
    filter_sizes = [3, 4, 5]
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            # W 是filter 矩阵
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]))
            # VALID padding意味着没有对句子的边缘进行padding
            conv = tf.nn.conv2d(embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID")
            # h 是对卷积结果进行非线性转换之后的结果
            h = tf.nn.relu(tf.nn.bias_add(conv, b))
            pooled = tf.nn.max_pool(h, ksize=[1, input_size-filter_size+1, 1, 1], strides=[1, 1, 1, 1], padding="VALID")
            pooled_outputs.append(pooled)

    num_filters_total = num_filters*len(filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    with tf.name_scope("dropout"):
        h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)
    with tf.name_scope("output"):
        W = tf.get_variable("W", shape=[num_filters_total, num_classes], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[num_classes]))
        output = tf.nn.xw_plus_b(h_drop, W, b)
    return output

# 模型训练
def train_neural_network(train_x, train_y):
    output = neural_network(vocab_size)
    num_batch = len(train_x) // batch_size
    optimizer = tf.train.AdamOptimizer(1e-3)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Y))
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(201):
            for i in range(num_batch):
                batch_x = train_x[i * batch_size: (i + 1) * batch_size]
                batch_y = train_y[i * batch_size: (i + 1) * batch_size]
                _, loss_ = sess.run([train_op, loss], feed_dict={X: batch_x, Y: batch_y, dropout_keep_prob: 0.5})
                print(e, i, loss_)
            # 保存模型
            if e % 50 == 0:
                saver.save(sess, r'.\train_model\name2sex.model', global_step=e)

# 计算准确率
def evaluation(test_x, test_y):
    output = neural_network(vocab_size)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 恢复前一次训练
        ckpt = tf.train.get_checkpoint_state(r'.\train_model')
        if ckpt != None:
            print(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("没找到模型")
        total_rate = 0
        test_batch = len(test_x) // batch_size
        count = 0
        for i in range(test_batch):
            batch_x = test_x[i * batch_size: (i + 1) * batch_size]
            batch_y = test_y[i * batch_size: (i + 1) * batch_size]
            prediction = sess.run(output, feed_dict={X: batch_x, dropout_keep_prob: 1.0})
            correct_prediction = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            if i % 50 == 0:
                accuracy_rate = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y, dropout_keep_prob: 1.0})
                print(accuracy_rate)
                total_rate += accuracy_rate
                count += 1
        # 92%
        print("the average accuracy is: ", total_rate/count)

# 根据姓名预测性别
def detect_gender(name_list):
    x = []
    for name in name_list:
        name_v = []
        for word in name:
            name_v.append(vocab.get(word))
        while len(name_v) < max_name_length:
            name_v.append(0)
        x.append(name_v)

    output = neural_network(vocab_size)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 恢复前一次训练
        ckpt = tf.train.get_checkpoint_state(r'.\train_model')
        if ckpt != None:
            print(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("没找到模型")

        x_pre = tf.argmax(output, 1)
        res = sess.run(x_pre, {X:x, dropout_keep_prob:1.0})
        for m, n in enumerate(name_list):
            print(n, '女' if res[m] == 0 else '男')

if __name__ == '__main__':
    name_vec, gender, vocab_size, vocab = dataproprecessing()
    X_train, X_test, y_train, y_test = train_test_split(name_vec, gender, test_size=0.2)
    input_size = max_name_length
    num_classes = 2
    batch_size = 64
    X = tf.placeholder(tf.int32, [None, input_size])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    dropout_keep_prob = tf.placeholder(tf.float32)
    # train_neural_network(X_train, y_train)
    evaluation(X_test, y_test)
    # detect_gender(["叶世强", "李冰冰", "王大锤", "司徒道"])

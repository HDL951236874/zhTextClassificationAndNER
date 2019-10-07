import tensorflow as tf
import numpy as np
import config

class TextCNN():
    def __init__(self,
                 seq_len,
                 train_epochs,
                 learning_rate,
                 embed_size,
                 batch_size,
                 class_num,
                 filter_size,
                 filter_num,
                 using_download = False):
        self.seq_len = seq_len
        self.train_epochs = train_epochs
        self.learning_rate = learning_rate

        self.embed_size = embed_size
        self.batch_size = batch_size
        self.class_num = class_num
        self.filter_size = filter_size
        self.filter_num = filter_num

        self.make_vocab()
        self.train_x, self.train_y = self.make_train_data()
        self.using_sownload = using_download

        self.input_x = tf.placeholder(tf.int32,[self.batch_size,self.seq_len])
        self.input_y = tf.placeholder(tf.float32, [self.batch_size, self.class_num])


        with tf.name_scope("embedding"):
            self.embed = tf.Variable(tf.random_uniform([self.vocab_size,self.embed_size],-0.5,0.5))
            self.embed_ = tf.nn.embedding_lookup(self.embed, self.input_x)
            self.embedding = tf.expand_dims(self.embed_,-1)


        with tf.name_scope("COV"):
            pool_outputs = []
            for size in self.filter_size:
                # the conv_layer's shape = [wid, length, channel_num, filter_num]
                filer_shape = [size, self.embed_size, 1, self.filter_num]
                W = tf.Variable(tf.truncated_normal(filer_shape, stddev=0.1))
                b = tf.Variable(tf.constant(0.1, shape=[self.filter_num]))

                conv =tf.nn.conv2d(self.embedding,
                                   W,
                                   strides=[1,1,1,1],
                                   padding='VALID')
                h = tf.nn.relu(tf.nn.bias_add(conv,b))
                pool = tf.nn.max_pool(h,
                                      ksize = [1, self.seq_len - size +1, 1, 1],
                                        strides=[1,1,1,1],
                                      padding='VALID'
                                      )
                pool_outputs.append(pool)
            self.filter_num_total = len(self.filter_size)*self.filter_num
            h_pool = tf.concat(pool_outputs, self.filter_num)
            self.h_pool_flat = tf.reshape(h_pool, [-1, self.filter_num_total])  # [batch_size, ]

        with tf.name_scope("full_connected"):
            W = tf.Variable(tf.random_normal([self.filter_num_total,self.class_num],-0.5,0.5))
            b = tf.Variable(tf.constant(1.0, shape = [self.class_num]))
            self.output =tf.nn.xw_plus_b(self.h_pool_flat, W, b)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.input_y))

        self.optim =tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        with tf.name_scope("pred"):
            self.predict = tf.argmax(tf.nn.softmax(self.output),1)


    def make_vocab(self):
        vocab = []
        with open(config.data_path + '/TextCNN/train.txt') as f:
            for line in f:
                for char in line.split(' ')[1]:
                    if char not in vocab:
                        vocab.append(char)
        self.vocab = vocab
        self.vocab_size = len(vocab)+1

    def make_train_data(self):
        train_x = []
        train_y = []
        with open(config.data_path + '/TextCNN/train.txt') as f:
            for line in f:
                line = line.split(' ')
                # train_x.append(line[1])
                list = []
                for char in line[1]:
                    list.append(self.vocab.index(char))
                while len(list)<self.seq_len:
                    list.append(self.vocab_size-1)
                train_x.append(list)
                train_y.append(line[0])
        return train_x, train_y

    def get_batch(self):
        input_x = []
        input_y = []
        random_index = np.random.choice(range(len(self.train_x)),self.batch_size, replace = False)

        for index in random_index:
            input_x.append(self.train_x[index])
            input_y.append(self.train_y[index])

        return input_x, input_y

if __name__ == "__main__":
    model = TextCNN(seq_len = 11,
                 train_epochs = 1000,
                 learning_rate = 0.001,
                 embed_size = 50,
                 batch_size = 2,
                 class_num = 2,
                    filter_size=[1,2,3,4],
                filter_num=2)


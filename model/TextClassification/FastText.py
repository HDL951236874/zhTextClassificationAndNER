import tensorflow as tf
import numpy as np
import config
import os

class FastText():
    def __init__(self,seq_length,embed_size,train_epochs,batch_size,
                 drop_out_rate, class_num, learning_rate):
        self.seq_length = seq_length
        self.embed_size = embed_size
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.drop_out_rate = drop_out_rate
        self.class_num = class_num
        self.learning_rate = learning_rate
        self.make_vocab()
        self.train_x, self.train_y = self.make_train_data()

        self.input_x = tf.placeholder(tf.int32,[self.batch_size,self.seq_length])
        self.input_y = tf.placeholder(tf.float32,[self.batch_size])

        with tf.name_scope("embedding"):
            self.embed = tf.Variable(tf.random_uniform([self.vocab_size,self.embed_size],-0.5,0.5))
            tf.summary.histogram("embed",self.embed)
            self.embedding = tf.nn.embedding_lookup(self.embed,self.input_x)

        with tf.name_scope("drop_out"):
            self.drop_out = tf.nn.dropout(self.embedding,self.drop_out_rate)

        with tf.name_scope("mean"):
            self.mean = tf.reduce_mean(self.drop_out,axis=1)

        with tf.name_scope("output"):
            output = tf.layers.dense(self.mean, self.class_num)
            self.output = tf.nn.softmax(output)

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(-tf.reduce_sum(self.input_y*tf.log(self.output),reduction_indices=[1]))
            tf.summary.scalar('loss', self.loss)
        self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def make_vocab(self):
        vocab = []
        with open(config.data_path + '/FastText/train.txt') as f:
            for line in f:
                for char in line.split(' ')[1]:
                    if char not in vocab:
                        vocab.append(char)
        self.vocab = vocab
        self.vocab_size = len(vocab)+1

    def make_train_data(self):
        train_x = []
        train_y = []
        with open(config.data_path + '/FastText/train.txt') as f:
            for line in f:
                line = line.split(' ')
                # train_x.append(line[1])
                list = []
                for char in line[1]:
                    list.append(self.vocab.index(char))
                while len(list)<self.seq_length:
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

    def train(self):

        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)
        merge = tf.summary.merge_all()
        writer = tf.summary.FileWriter("logs/", sess.graph)

        for _ in range(self.train_epochs):
            inputs, label = self.get_batch()
            # inputs = np.array(inputs)
            # label = np.array(label)
            pre,loss,output_label = sess.run([self.optim,self.loss,self.output],feed_dict={self.input_x.name : inputs, self.input_y.name: label})

            if _%10 == 0:
                print(loss)
                # output_label = sess.run(self.output)
                print("output"+ ' ' + str(output_label))
                res = sess.run(merge,feed_dict={self.input_x.name : inputs, self.input_y.name : label})
                writer.add_summary(res,_)
def make_train_file():
    with open(config.data_path + '/FastText/train.txt','w') as f:
        f.write("0 今天天气不错\n")
        f.write("0 今天我很开心\n")
        f.write("0 这个电影很好看\n")
        f.write("0 我很喜欢你\n")
        f.write("1 今天我不是很开心\n")
        f.write("1 我不是很喜欢做这件事\n")
        f.write("1 今天有点烦\n")
        f.write("1 我有点烦你")

        f.close()
    with open(config.data_path + '/FastText/test.txt','w') as f:
        f.write("我不喜欢你\n")
        f.close()

if __name__ =="__main__":
    # make_train_file()
    model = FastText(seq_length = 11,embed_size=2,train_epochs=1000, batch_size=2,
                 drop_out_rate=0.1, class_num=2, learning_rate=0.001)

    model.train()
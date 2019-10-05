import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']


class word2vec():
    def __init__(self,embedding_size,batch_size,training_epochs,num_sampled,data):
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.input_x = tf.placeholder(tf.int32,[self.batch_size])
        self.input_y = tf.placeholder(tf.int32,[self.batch_size, 1])

        self.data = data
        self.vocab = self.make_vocab(self.data)
        self.vocab_size = len(self.vocab)

        self.train_x, self.train_y = self.make_train_data(self.data)

        self.training_epochs = training_epochs

        with tf.name_scope("embedding"):
            self.embeds = tf.Variable(tf.random_uniform([self.vocab_size,self.embedding_size],-0.05,0.05))
            # normal_embedding = tf.nn.l2_normalize(embeds,1)
            self.embedding = tf.nn.embedding_lookup(self.embeds, self.input_x)

        with tf.name_scope("output"):
            self.out_put = tf.nn.softmax(self.embedding)

        with tf.name_scope("loss"):
            nce_weights = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0))
            nce_biases = tf.Variable(tf.zeros([self.vocab_size]))

            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(nce_weights, nce_biases, self.input_y, self.embedding, self.num_sampled, self.vocab_size))

        self.optim = tf.train.AdamOptimizer(0.001).minimize(self.loss)


    def make_train_data(self, data):
        xs, ys = [], []
        for string in data:
            for index in range(1,len(string)-1):

                xs.append(self.vocab.index(string[index-1]))
                xs.append(self.vocab.index(string[index+1]))
                ys.append(self.vocab.index(string[index]))
                ys.append(self.vocab.index(string[index]))

        return xs, ys


    def make_vocab(self,data):
        vocab = []
        for string in data:
            for index in string:
                if index not in vocab:
                    vocab.append(index)

        return vocab

    def get_batch(self,train_x,train_y,size):
        random_inputs = []
        random_labels = []
        random_index = np.random.choice(range(len(train_x)), size, replace=False)
        for i in random_index:
            random_inputs.append(train_x[i])
            random_labels.append([train_y[i]])

        return random_inputs, random_labels

    def train(self):
        sess = tf.Session()
        init =tf.global_variables_initializer()
        sess.run(init)

        for _ in range(self.training_epochs):
            inputs, labels = self.get_batch(self.train_x,self.train_y,self.batch_size)
            pre, loss = sess.run([self.optim,self.loss],feed_dict={self.input_x.name: inputs, self.input_y.name:labels})

            if _%10 == 0:
                print('Epoch:', '%04d' % (_ + 1), 'cost =', '{:.6f}'.format(loss))

        trained_embeddings = sess.run(self.embeds)

        for i, label in enumerate(self.vocab):
            x, y = trained_embeddings[i]
            plt.scatter(x, y)
            plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
        plt.show()
        print(trained_embeddings)

if __name__ == "__main__":
    embedding_size = 2
    batch_size = 5

    test = ["今天天气不错","今天我很开心","但是有些事情不知道心里怎么说"]

    model = word2vec(embedding_size=embedding_size,
                     batch_size=batch_size,training_epochs=50,num_sampled=1,data=test)

    model.train()
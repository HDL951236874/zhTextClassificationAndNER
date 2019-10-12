import tensorflow as tf
import numpy as np
import config
import os
import re
import random


class Bert():
    def __init__(self,
                 seq_len,
                 train_epochs,
                 learning_rate,
                 hidden_size,
                 batch_size,
                 class_num,
                 max_prd,
                 head_num):
        self.seq_len = seq_len
        self.train_epochs = train_epochs
        self.learning_rate = learning_rate

        self.hidden_size = hidden_size
        self.head_num = head_num
        self.batch_size = batch_size
        self.class_num = class_num
        self.segment_num = 2
        self.max_prd = max_prd
        self.make_vocab_and_train_data()

        self.input_x = tf.placeholder(tf.int32, [self.batch_size, self.seq_len])
        self.input_pos = tf.placeholder(tf.int32, [self.batch_size, self.seq_len])
        self.input_seg_id = tf.placeholder(tf.int32, [self.batch_size, self.seq_len])
        self.input_mask = tf.placeholder(tf.int32, [self.batch_size, self.seq_len])
        self.input_att_mask = tf.placeholder(tf.int32, [self.batch_size, self.seq_len])

        with tf.name_scope("embedding"):
            token_embeds = tf.Variable(tf.random_normal([self.vocab_size, self.hidden_size]))
            pos_embeds = tf.Variable(tf.random_normal([self.seq_len, self.hidden_size]))
            segment_embeds = tf.Variable(tf.random_normal([self.segment_num, self.hidden_size]))
            token_embedding = tf.nn.embedding_lookup(token_embeds, self.input_x)
            pos_embedding = tf.nn.embedding_lookup(pos_embeds, self.input_pos)
            segment_embedding = tf.nn.embedding_lookup(segment_embeds, self.input_seg_id)

            embed_after_nor = self.LayerNormalization()(token_embedding + pos_embedding + segment_embedding)

            x = tf.nn.dropout(embed_after_nor, keep_prob=0.8)

        with tf.name_scope("multi-head self att"):
            for _ in range(12):
                x = self.Bert_self_mulihead_attention_layer(x,x,x)




    class LayerNormalization(tf.layers.Layer):
        def __init__(self, gamma=1, beta=0):
            super().__init__()
            self.gamma = gamma
            self.beta = beta
            self.eps = 10e-5

        def build(self, input_shape):
            self.built = True

        def call(self, inputs, **kwargs):
            x_mean = np.mean(inputs, axis=(1, 2), keepdims=True)
            x_var = np.var(inputs, axis=(1, 2), keepdims=True)
            x_normalized = (inputs - x_mean) / np.sqrt(x_var + self.eps)
            res = x_normalized * self.gamma + self.beta
            return res

    def Bert_layer(self, input):
        after_atten = self.Bert_self_mulihead_attention_layer(input, input, input)
        after_output = self.Bert_output_layer(after_atten)
        after_poswise = self.Bert_intermediate_layer(after_output)

    def Bert_self_mulihead_attention_layer(self, K, Q, V):

        KW, KB = tf.Variable(tf.random_normal([self.hidden_size, self.hidden_size])) \
            , tf.Variable(tf.random_normal([1, self.hidden_size]))
        QW, QB = tf.Variable(tf.random_normal([self.hidden_size, self.hidden_size])), \
                 tf.Variable(tf.random_normal([1, self.hidden_size]))
        VW, VB = tf.Variable(tf.random_normal([self.hidden_size, self.hidden_size])) \
            , tf.Variable(tf.random_normal([1, self.hidden_size]))

        key = tf.matmul(K, KW) + KB
        query = tf.matmul(Q, QW) + QB
        value = tf.matmul(V, VW) + VB

        tf.transpose(key, [self.batch_size, self.head_num, self.hidden_size / self.head_num, self.seq_len])
        tf.transpose(query, [self.batch_size, self.head_num, self.hidden_size / self.head_num, self.seq_len])
        tf.transpose(value, [self.batch_size, self.head_num, self.hidden_size / self.head_num, self.seq_len])

        att_mask = tf.transpose(self.input_att_mask)

        context = self.ScaleDotProductAtt(query,key,value,att_mask)
        tf.transpose(context, [self.batch_size, self.seq_len, self.hidden_size])

        output =tf.layers.Dense(self.hidden_size,self.hidden_size)(context)

        output = self.LayerNormalization()(output)

        output = self.Bert_intermediate_layer(output)

        return output

    # this is the layer for the poswise self attention
    def Bert_intermediate_layer(self, input):
        x = tf.layers.dense(4*self.hidden_size,input)
        x = self.gelu(x)
        x = tf.layers.dense(self.hidden_size,x)
        x = tf.nn.dropout(x)
        return x

    # this is the gelu activite function
    def gelu(self,input_tensor):
        cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
        return input_tensor * cdf


    def ScaleDotProductAtt(self, Q, K, V, att_mask):
        mask = tf.constant(-1e9,shape=[att_mask.shape])
        score = tf.matmul(Q, tf.transpose(K, [self.batch_size, self.head_num, self.seq_len,
                                              self.hidden_size / self.head_num])) / tf.sqrt(
            self.hidden_size / self.head_num)
        score_after_mask = tf.where(att_mask == 1, mask, score)
        attn = tf.nn.softmax(score_after_mask,dim=-1)
        return tf.matmul(attn,V)

    def make_vocab_and_train_data(self):
        text = (
            'Hello, how are you? I am Romeo.\n'
            'Hello, Romeo My name is Juliet. Nice to meet you.\n'
            'Nice meet you too. How are you today?\n'
            'Great. My baseball team won the competition.\n'
            'Oh Congratulations, Juliet\n'
            'Thanks you Romeo'
        )
        self.sentences = re.sub("[.,!?\\-]", '', text.lower()).split('\n')  # filter '.', ',', '?', '!'
        word_list = list(set(" ".join(self.sentences).split()))
        word_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
        for i, w in enumerate(word_list):
            word_dict[w] = i + 4
        self.number_dict = {i: w for i, w in enumerate(word_dict)}
        vocab_size = len(word_dict)
        token_list = list()
        for sentence in self.sentences:
            arr = [word_dict[s] for s in sentence.split()]
            token_list.append(arr)
        self.vocab_size = vocab_size
        self.word_dict = word_dict
        self.train_data = token_list

    def get_batch(self):
        batch = []
        positive = negative = 0
        while positive < self.batch_size and negative < self.batch_size:
            tokens_a_index, tokens_b_index = random.randrange(len(self.sentences)), random.randrange(
                len(self.sentences))
            tokens_a, tokens_b = self.train_data[tokens_a_index], self.train_data[tokens_b_index]
            input_ids = [self.word_dict["[CLS]"]] + tokens_a + [self.word_dict["SEP"]] + tokens_b + [
                self.word_dict["[SEP]"]]
            segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)

            # MLM
            n_pre = min(self.max_prd, max(1, int(round(len(input_ids) * 0.15))))
            cand_maked_pos = [i for i, token in enumerate(input_ids)
                              if token != self.word_dict['[CLS]'] and token != self.word_dict['[SEP]']]
            random.shuffle(cand_maked_pos)
            masked_tokens, masked_pos = [], []
            for pos in cand_maked_pos[:n_pre]:
                masked_pos.append(pos)
                masked_tokens.append(input_ids[pos])
                if random.random() < 0.8:
                    input_ids[pos] = self.word_dict["MASK"]
                elif random.random() < 0.5:
                    index = random.randint(0, self.vocab_size - 1)
                    input_ids[pos] = self.number_dict[index]

            # zero paddding
            n_pad = self.seq_len - len(input_ids)
            input_ids.extend([0] * n_pad)
            segment_ids.extend([0] * n_pad)

            # token_masked padding
            if self.max_prd > n_pre:
                n_pad = self.max_prd - n_pre
                masked_tokens.extend([0] * n_pad)
                masked_pos.extend([0] * n_pad)

            if tokens_a_index + 1 == tokens_b_index and positive < self.batch_size / 2:
                batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True])  # IsNext
                positive += 1
            elif tokens_a_index + 1 != tokens_b_index and negative < self.batch_size / 2:
                batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False])  # NotNext
                negative += 1

            batch_size, seq_len = len(batch), len(batch[0][0])
            # att_mask = [1 if z == 0 else 0 for z in [y for y in [x[0] for x in batch]]]
            att_mask_ids = []
            for ids in [x[0] for x in batch]:
                att_mask_id = []
                for index in ids:
                    att_mask_id += [0] if index != 0 else 1
                att_mask_ids.append(att_mask_id)

            batch.insert(4, att_mask_ids)
            return batch

    def get_attention_pad_mask(self):
        pass


if __name__ == "__main__":
    model = Bert(seq_len=100,
                 train_epochs=10,
                 learning_rate=0.00001,
                 hidden_size=768,
                 batch_size=2,
                 class_num=2,
                 max_prd=3,
                 head_num=12)
    # model.make_vocab_and_train_data()
    # x = np.random.rand(3,3,3)
    # layer = model.LayerNormalization()
    # y = layer(x)
    # print(y)
    print(1)

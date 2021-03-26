
def test():
    return "hello world!"


def load_image(image_path):
    print("Func: load image.")
    img = tf.io.read_file(image_path)  # 转换为string类型的tensor
    img = tf.image.decode_jpeg(img, channels=3)  # 将图片解码为unit8的tensor
    img = tf.image.resize(img, (299, 299))  # resizing - img应当符合inception规定的大小
    img = tf.keras.applications.inception_v3.preprocess_input(
        img)  # normalize the image so that it contains pixels in [-1, 1]

    return img, image_path


# 建立 inception-v3 model
# iv3的最后一个卷积层shape为(8*8*2048)
# 将每一个图片馈入网络中, 将结果向量存在词典中 (image_name --> feature_vector)

image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
new_input = image_model.input  # 获取模型的input - return一个tensor
hidden_layer = image_model.layers[-1].output  # hidden_layer为iv3的最后一个层的输出

# 根据输入输出建立model
image_features_extract_model = tf.keras.Model(inputs=new_input, outputs=hidden_layer)



# In[6]:

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()  # 要用到三个FC网络
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)  # 要与features有相同的位数 (a, b, c)

        # attention_hidden_layer shape == (batch_size, 64, units)
        # - Calculating the Alignment vector (score)
        # - calculated between the previous decoder hidden state and each of the encoder’s hidden states
        # - 度量环境向量与当前输入向量的相似性；找到当前环境下，应该 focus 哪些输入信息；
        # - features: 输入向量. hwta: 环境向量
        # - alignment vector的每一个值都是原序列中对应单词的分数(或概率)
        #### attention_hidden_layer = (tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))) # 要使二者有相同shape

        # score shape == (batch_size, 64, 1)
        # This gives you an unnormalized score for each image feature.
        # - 一个前向传播网络FNN, 计算
        #### score = self.V(attention_hidden_layer)
        score = self.V((tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))))  # score=eij, ( hj, st-1 )

        # attention_weights shape == (batch_size, 64, 1)
        # - alignment function：计算 attention weight，通常都使用 softmax 进行归一化；
        attention_weights = tf.nn.softmax(score, axis=1)  # atj

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)  # sum(atj*hj)

        return context_vector, attention_weights


# In[7]:


class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim)
        # shape after fc == (batch_size, 64, embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

        # Since you have already extracted the features and dumped it using pickle
        # This encoder passes those features through a Fully connected layer


# In[8]:


class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)  # 实际单词表的个数

        self.attention = BahdanauAttention(self.units)

    @tf.function
    def call(self, x, features, hidden):
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        # - x是输入, 进行词嵌入
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        # - 降维处理
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

    def get_config(self):
        return {"features": self.features, "hidden": self.hidden}


# In[9]:


encoder_2 = CNN_Encoder(embedding_dim)
decoder_2 = RNN_Decoder(embedding_dim, units, vocab_size)

# In[10]:


encoder_2.load_weights('./model_weights/encoder_weights')
decoder_2.load_weights('./model_weights/decoder_weights')


# In[11]:


# 封装 注意加载关联项
# - encoder, decoder, load_image(),   # 注意封装
def evaluate_2(image):
    # attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder_2.reset_state(batch_size=1)  # 每次evaluate都要reset隐藏层hidden的状态

    temp_input = tf.expand_dims(load_image(image)[0], 0)  # 将image用load_处理(包括解码, tensor, resize, normalize), 然后升维
    img_tensor_val = image_features_extract_model(temp_input)  # 使用interception模型导出features
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))  # 维度处理

    features = encoder_2(img_tensor_val)  # 用自定义CNN进行编码处理

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)  # decoder的预输入
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder_2(dec_input, features, hidden)
        # predictions 是decoder处理完图像后, 获取的可能的输出 (是一个概率分布)
        # attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()  # 获取概率最高的那个单词的id
        result.append(tokenizer.index_word[predicted_id])  # 根据id找单词

        if tokenizer.index_word[predicted_id] == '<end>':  # 当遇到结束符, 则返回
            return result  # , attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)  # ???
        # print(dec_input.shape())

    # attention_plot = attention_plot[:len(result), :]
    return result  # , attention_plot


# In[16]:
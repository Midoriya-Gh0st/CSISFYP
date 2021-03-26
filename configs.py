import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
import joblib
 

APP_SITE = ""
UPLOAD_FOLDER = 'static/upload/'
ALLOWED_EXTENSIONS = ['jpg']
UPLOAD_ID = 0  # image
AUDIO_ID = 0  # audio

# In[15]:

BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
top_k = 5000
vocab_size = top_k + 1
max_length = 46
tokenizer = joblib.load('./model_weights/tokenizer.pkl')


def test():
    return "hello world!"


def load_image(image_path):
    # print("Func: load image.")
    img = tf.io.read_file(image_path)  # transfer to string tensor
    img = tf.image.decode_jpeg(img, channels=3)  # decode image as unit8 tensor
    img = tf.image.resize(img, (299, 299))  # resizing - img should meet the size adopted in Inception-v3
    # normalize the image so that it contains pixels in [-1, 1]
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


# Build inception-v3 model
# The shape of the last convolutional layer of iv3 is (8*8*2048)
# Feed each picture into the network and store the result vector in the dictionary (image_name --> feature_vector)

image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
new_input = image_model.input  # Get the input of the model-return a tensor
hidden_layer = image_model.layers[-1].output  # hidden_layer is the output of the last layer of iv3

# Build a model based on input and output.
# Model used to extract image features.
image_features_extract_model = tf.keras.Model(inputs=new_input, outputs=hidden_layer)


# In[6]:

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()  # will use 3 dense models
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)  # must have the same number of digits as features (a, b, c)

        # - calculating the alignment vector (score)
        # - calculated between the previous decoder hidden state and each of the encoder’s hidden states
        # - measure the similarity between the environment vector and the current input vector; 
        # - find out which input information should be focused in the current environment;
        # - features: input vector. hwta: environment vector
        # - each value of the alignment vector is the score (or probability) of the corresponding word in the original sequence
        # - attention_hidden_layer = (tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))) # same shape

        # - score shape == (batch_size, 64, 1) 
        # - a forward propagation network FNN, calculated
        # - score = self.V(attention_hidden_layer)
        score = self.V((tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))))  # score=eij, ( hj, st-1 )

        # - attention_weights shape == (batch_size, 64, 1)
        # - alignment function：compute attention weight，commonly use softmax to normalise.
        attention_weights = tf.nn.softmax(score, axis=1)  # atj

        # - context_vector shape after sum == (batch_size, hidden_size)
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

        # have extracted the features and dumped it using pickle
        # encoder passes those features through a FC layer


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
        self.fc2 = tf.keras.layers.Dense(vocab_size)  # number of actual word in vocabulary

        self.attention = BahdanauAttention(self.units)

    @tf.function
    def call(self, x, features, hidden):
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        # - x is the input, start to word embedding
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        # - dimensiona reduction
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


# Encapsulation: Pay attention to loading related items
def evaluate_2(image):
    # attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder_2.reset_state(batch_size=1)  # must reset the hidden state of the hidden layer each time

    temp_input = tf.expand_dims(load_image(image)[0], 0)  # process the image with load_(including decoding, tensor, resize, normalize), then                                                                                                                                                                                升维
    img_tensor_val = image_features_extract_model(temp_input)  # use Interception-v3 model extract image features                          
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))  # dimension processing

    features = encoder_2(img_tensor_val)  # use custom CNN for encoding

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)  # input of decoder
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder_2(dec_input, features, hidden)
        # predictions is the possible output obtained after the decoder processes the image (it is a probability distribution)
        # attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()  # get the word id with the highest probability
        result.append(tokenizer.index_word[predicted_id])  # find word by id

        if tokenizer.index_word[predicted_id] == '<end>':  # when the end character is encountered, return
            return result

        dec_input = tf.expand_dims([predicted_id], 0)
        # print(dec_input.shape())

    return result
import os
from abc import ABC

import tensorflow as tf
from joblib import load as jload

print("[0.-]. env CUDA disabled successfully.")

APP_SITE = ""
UPLOAD_FOLDER = 'static/upload/'
ALLOWED_EXTENSIONS = ['jpg']
UPLOAD_ID = 0  # image
AUDIO_ID = 0  # audio

BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
top_k = 5000
vocab_size = top_k + 1
max_length = 46

tokenizer = jload('./model_weights/tokenizer.pkl')
print("[0.-]. Tokenizer (words analyser) loaded.")

# Build inception-v3 model
image_model = tf.keras.models.load_model("model_weights/image_model_pre.h5")
new_input = image_model.input  # Get the input of the model-return a tensor
hidden_layer = image_model.layers[-1].output  # hidden_layer is the output of the last layer of iv3
print("[0.-]. Inception-V3 model loaded.")

image_features_extract_model = tf.keras.Model(inputs=new_input, outputs=hidden_layer)

""" program function & class """


def load_image(image_path):
    # print("Func: load image.")
    img = tf.io.read_file(image_path)  # transfer to string tensor
    img = tf.image.decode_jpeg(img, channels=3)  # decode image as unit8 tensor
    img = tf.image.resize(img, (299, 299))  # resizing - img should meet the size adopted in Inception-v3
    # normalize the image so that it contains pixels in [-1, 1]
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


class BahdanauAttention(tf.keras.Model, ABC):
    def __init__(self, a_units):
        super(BahdanauAttention, self).__init__()  # will use 3 dense models
        self.W1 = tf.keras.layers.Dense(a_units)
        self.W2 = tf.keras.layers.Dense(a_units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)  # must have the same number of digits as features (a, b, c)
        score = self.V((tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))))  # score=eij, ( hj, st-1 )

        attention_weights = tf.nn.softmax(score, axis=1)  # atj

        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)  # sum(atj*hj)

        return context_vector, attention_weights


class CNN_Encoder(tf.keras.Model, ABC):
    def __init__(self, p_embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(p_embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


class RNN_Decoder(tf.keras.Model):
    def __init__(self, p_embedding_dim, p_units, p_vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = p_units
        self.embedding = tf.keras.layers.Embedding(p_vocab_size, p_embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units, return_sequences=True, return_state=True,
                                    recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(p_vocab_size)  # number of actual word in vocabulary
        self.attention = BahdanauAttention(self.units)

    @tf.function
    def call(self, x, features, hidden):
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)

        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)
        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

    def get_config(self):
        return {"features": self.features, "hidden": self.hidden}


# Encapsulation: Pay attention to loading related items
def evaluate_2(image, encoder, decoder, tok):
    # attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)  # must reset the hidden state of the hidden layer each time

    temp_input = tf.expand_dims(load_image(image)[0],
                             0)  # process the image with load_(including decoding, tensor, resize, normalize), then                                                                                                                                                                                升维
    img_tensor_val = image_features_extract_model(temp_input)  # use Interception-v3 model extract image features
    img_tensor_val = tf.reshape(img_tensor_val,
                             (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))  # dimension processing

    features = encoder(img_tensor_val)  # use custom CNN for encoding

    dec_input = tf.expand_dims([tok.word_index['<start>']], 0)  # input of decoder
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()  # get the word id with the highest probability
        result.append(tok.index_word[predicted_id])  # find word by id

        if tok.index_word[predicted_id] == '<end>':  # when the end character is encountered, return
            return result

        dec_input = tf.expand_dims([predicted_id], 0)

    return result


encoder_2 = CNN_Encoder(embedding_dim)
decoder_2 = RNN_Decoder(embedding_dim, units, vocab_size)

encoder_2.load_weights('./model_weights/encoder_weights')
decoder_2.load_weights('./model_weights/decoder_weights')
print("[0.-]. Encoder(CNN) & Decoder(RNN) created.")

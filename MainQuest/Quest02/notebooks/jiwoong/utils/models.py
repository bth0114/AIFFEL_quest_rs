from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, Input, Layer
import tensorflow.keras.backend as K
import fasttext
import numpy as np

def build_model_baseline(num_words, input_max_length, embedding_dim):
  vocab_size = num_words + 1  # 패딩

  inputs = Input(shape=(input_max_length,))
  embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)
  lstm = Bidirectional(LSTM(64))(embedding)
  dropout = Dropout(0.5)(lstm)
  outputs = Dense(5, activation='softmax')(dropout)
  model = Model(inputs=inputs, outputs=outputs)

  return model

class AttentionLayer(Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)
    
def build_model_attention(num_words, input_max_length, embedding_dim):
  vocab_size = num_words + 1  # 패딩

  inputs = Input(shape=(input_max_length,))
  embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)
  lstm = Bidirectional(LSTM(64, return_sequences=True))(embedding)
  attention = AttentionLayer()(lstm) # 어텐션 추가
  dropout = Dropout(0.5)(attention)
  outputs = Dense(5, activation='softmax')(dropout)
  model = Model(inputs=inputs, outputs=outputs)

  return model

# 임베딩 행렬 생성 함수
def build_embedding_matrix(ft_model, tokenizer, num_words, embedding_dim):
    vocab_size = num_words + 1 # 패딩
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for word, idx in tokenizer.word_index.items():
        if idx >= vocab_size:
            continue
        embedding_vector = ft_model.get_word_vector(word)
        if embedding_vector is not None:
            embedding_matrix[idx] = embedding_vector
    return embedding_matrix

# 워드 임베딩을 파인튜닝 하지 않은 모델
def build_model_pretrained_embedding(tokenizer, num_words, input_max_length, embedding_dim):

    # 사전학습된 임베딩 불러오기
    ft = fasttext.load_model('/content/drive/MyDrive/Colab Notebooks/Aiffel/AIFFEL_DLThon_DKTC_online13/notebooks/jiwoong/data/cc.ko.100.bin')
    embedding_matrix = build_embedding_matrix(ft, tokenizer, num_words, embedding_dim)

    vocab_size = embedding_matrix.shape[0]

    inputs = Input(shape=(input_max_length,))
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                          weights=[embedding_matrix],
                          trainable=True)(inputs)
    lstm = Bidirectional(LSTM(64))(embedding)
    dropout = Dropout(0.5)(lstm)
    outputs = Dense(5, activation='softmax')(dropout)
    model = Model(inputs=inputs, outputs=outputs)

    return model
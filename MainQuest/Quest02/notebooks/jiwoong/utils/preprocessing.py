import re
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 구두점을 제거하는 함수
def remove_punctuation(texts):
    result = []
    for text in texts:
        # ?! 제외한 모든 특수문자 제거
        text = re.sub(r'[^\w\s?!]', '', text.replace('\n', ' '))
        result.append(text)
    return result

# 명사, 동사, 형용사, 부사, 숫자를 기준으로 토큰화해주는 함수
def pos_tagging(texts):
    okt = Okt()

    result = []
    for text in texts:
        filtered_words = [word for word, pos in okt.pos(text) if pos in [
            'Noun', 'Verb', 'Adjective', 'Adverb',  # 기본 품사
            'Number'                                # 금전 관련 숫자 정보
        ]]
        result.append(filtered_words)  # 각 문장별로 필터링된 단어들을 리스트로 저장

    return result

# 토크나이저 생성 함수
def create_tokenizer(texts, num_words=None): # num_words = 인코딩 시 사용할 단어 개수
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts) # 어휘사전 정의

    return tokenizer

# 모든 전처리 함수. 구두점 제거, 품사기준 토큰화, 정수 인코딩, 패딩
def preprocessing(texts, tokenizer, padded_max_len):
  removed_punctuation_texts = remove_punctuation(texts)
  pos_tagged_texts = pos_tagging(removed_punctuation_texts)
  sequences = tokenizer.texts_to_sequences(pos_tagged_texts)
  padded_sequences = pad_sequences(sequences, maxlen=padded_max_len)

  return padded_sequences
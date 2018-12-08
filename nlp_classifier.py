import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
import tqdm

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras
from keras.layers import Input, Dense, Dropout, Lambda, Flatten, Concatenate
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical



will_use_gpu = spacy.prefer_gpu()
print(f"spaCy will use gpu: {will_use_gpu}")


# %%
# Data - load and explore
df_train = pd.read_csv("input/train.csv")
df_test = pd.read_csv("input/test.csv")


# %%
# Questions - extract texts
MAX_SEQUENCE_LENGTH = 60
BATCH_SIZE = 512
Q_FRACTION = 1
questions = df_train.sample(frac=Q_FRACTION)
question_texts = questions["question_text"].values
question_targets = questions["target"].values
print(f"Working on {len(questions)} questions")


# %% NLP tools
nlp = spacy.load("en_core_web_sm", disable=['parser'])
nlp.add_pipe(nlp.create_pipe('sentencizer'))
print(nlp.pipe_names)


# %%
# Find POS and NER tags
# Entity types from https://spacy.io/api/annotation#named-entities
pos_tags = nlp.tokenizer.vocab.morphology.tag_map.keys()
pos_tags_count = len(pos_tags)
entity_types = ["PERSON", "NORP", "FAC", "ORG", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "DATE",
                "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]
entity_types_count = len(entity_types)


pos_tokenizer = Tokenizer(num_words=pos_tags_count, lower=False)
pos_tokenizer.fit_on_texts(pos_tags)
default_filter_without_underscore = '!"#$%&()*+,-./:;<=>?@[\]^`{|}~'
entity_tokenizer = Tokenizer(num_words=entity_types_count, lower=False, oov_token='0',
                             filters=default_filter_without_underscore)
entity_tokenizer.fit_on_texts(list(entity_types))
entity_types_count = len(entity_tokenizer.index_word) + 1

def token_encoded_pos_getter(token):
    if token.tag_ in pos_tokenizer.word_index:
        return pos_tokenizer.word_index[token.tag_]
    else:
        return 0


def token_encoded_ent_getter(token):
    if token.ent_type_ in entity_tokenizer.word_index:
        return entity_tokenizer.word_index[token.ent_type_]
    else:
        return 0


spacy.tokens.token.Token.set_extension('encoded_pos', force=True, getter=token_encoded_pos_getter)
spacy.tokens.token.Token.set_extension('encoded_ent', force=True, getter=token_encoded_ent_getter)
spacy.tokens.doc.Doc.set_extension('encoded_pos', force=True, getter=lambda doc: [token._.encoded_pos for token in doc])
spacy.tokens.doc.Doc.set_extension('encoded_ent', force=True, getter=lambda doc: [token._.encoded_ent for token in doc])

pos_encodings = []
ent_encodings = []
for doc in tqdm.tqdm(nlp.pipe(question_texts, batch_size=100, n_threads=4), total=len(question_texts)):
    pos_encodings.append(doc._.encoded_pos)
    ent_encodings.append(doc._.encoded_ent)

pos_encodings = np.array(pos_encodings)
pos_encodings = pad_sequences(pos_encodings, maxlen=MAX_SEQUENCE_LENGTH)
pos_encodings = to_categorical(pos_encodings, num_classes=pos_tags_count)
# print(pos_encodings)

ent_encodings = np.array(ent_encodings)
ent_encodings = pad_sequences(ent_encodings, maxlen=MAX_SEQUENCE_LENGTH)
ent_encodings = to_categorical(ent_encodings, num_classes=entity_types_count)
# print(ent_encodings)


# %%
# Model and evaluation

def display_model_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.show()


# %%
ent_input = Input(shape=(MAX_SEQUENCE_LENGTH, entity_types_count), name="ent_input")
x_ent = Flatten()(ent_input)
x_ent = Dense(100)(x_ent)
x_ent = Dropout(0.5)(x_ent)
x_ent = Dense(10)(x_ent)

pos_input = Input(shape=(MAX_SEQUENCE_LENGTH, pos_tags_count), name="pos_input")
x_pos = Flatten()(pos_input)
x_pos = Dense(300)(x_pos)
x_pos = Dropout(0.5)(x_pos)
x_pos = Dense(20)(x_pos)

x = Concatenate()([x_ent, x_pos])
out = Dense(1, activation="sigmoid")(x)

model = Model(inputs=[pos_input, ent_input], outputs=out)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(
    x={"pos_input": pos_encodings, "ent_input": ent_encodings},
    y=question_targets,
    batch_size=512, epochs=10, verbose=1, validation_split=0.015)
display_model_history(history)

import numpy as np
import pandas as pd
import tqdm

import os
print(os.listdir("input"))


# %% Data - load and explore
df_train = pd.read_csv("input/train.csv")
df_test = pd.read_csv("input/test.csv")

# %% Data shapes
print(f"train: {df_train.shape}")
print(f"test: {df_test.shape}")


# %% Question NLP insights
%%time
import spacy
will_use_gpu = spacy.prefer_gpu()
print(will_use_gpu)

q_fraction = 0.05
questions = df_train.sample(frac=q_fraction)
question_texts = questions["question_text"].values
nlp = spacy.load("en_core_web_sm", disable=['parser'])
nlp.add_pipe(nlp.create_pipe('sentencizer'))
# Prefer the pipe() as described in https://spacy.io/usage/processing-pipelines#section-multithreading
docs = [doc for doc in tqdm.tqdm(nlp.pipe(question_texts, batch_size=1000, n_threads=2))]
# docs = [nlp(q) for q in tqdm.tqdm(questions)]



# %%
# Find POS and NER tags
# Entity types from https://spacy.io/api/annotation#named-entities

all_pos_tags = nlp.tokenizer.vocab.morphology.tag_map.keys()
tags_count = nlp.tokenizer.vocab.morphology.n_tags
entitiy_types = ["PERSON", "NORP", "FAC", "ORG", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]
entitiy_types_count = len(entitiy_types)

import keras
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer, one_hot, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

pos_tokenizer = Tokenizer(num_words=tags_count, lower=False)
pos_tokenizer.fit_on_texts(all_pos_tags)
default_filter_without_underscore = '!"#$%&()*+,-./:;<=>?@[\]^`{|}~'
ne_tokenizer = Tokenizer(num_words=entitiy_types_count, lower=False, oov_token='0', filters=default_filter_without_underscore)
ne_tokenizer.fit_on_texts(list(entitiy_types))

one_hot_tags = []
one_hot_ners = []
unique_tags = set()
unique_ents = set()

for doc in docs[3:4]:
    # print(doc)
    # for token in doc:
    #     unique_ents.add(token.ent_type_)
    #     unique_tags.add(token.tag_)
    #     if token.ent_type_ not in all_ners and token.ent_type_ is not "":
    #         print(f"{token} + {token.ent_type_}")

    # Working POS one-hot-encoding
    pos_tags = " ".join([token.tag_ for token in doc])
    pos_tags = pos_tokenizer.texts_to_sequences([pos_tags])
    pos_tags = pad_sequences(pos_tags, maxlen=60)
    pos_tags = to_categorical(pos_tags, num_classes=tags_count)

    # Working NER one-hot-encoding
    named_entities = " _ ".join([token.ent_type_ for token in doc])
    named_entities = ne_tokenizer.texts_to_sequences([named_entities])
    named_entities = pad_sequences(named_entities, maxlen=60)
    named_entities = to_categorical(named_entities, num_classes=entitiy_types_count)



# print(unique_ents)
# print(unique_tags)

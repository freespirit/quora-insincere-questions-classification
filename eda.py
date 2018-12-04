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

# %% Sincere examples
sincere = df_train.loc[df_train['target'] == 0]
sincere.head()

# %% Insincere examples
insincere = df_train.loc[df_train['target'] == 1]
insincere.head()



# %% Question NLP insights
# %%time
import spacy
q_fraction = 1
questions = df_train.sample(frac=q_fraction)
insincere_indices = []
sincere_indices = []
index = 0
for _, row in questions.iterrows():
    if row['target'] == 0:
        sincere_indices.append(index)
    else:
        insincere_indices.append(index)
    index += 1

question_texts = questions["question_text"].values
nlp = spacy.load("en_core_web_sm", disable=['parser'])
nlp.add_pipe(nlp.create_pipe('sentencizer'))
# Prefer the pipe() as described in https://spacy.io/usage/processing-pipelines#section-multithreading
docs = [doc for doc in tqdm.tqdm(nlp.pipe(question_texts, batch_size=1000, n_threads=2))]
# docs = [nlp(q) for q in tqdm.tqdm(questions)]


# %% Plot the words-in-question distrubution
import seaborn as sns
import matplotlib.pyplot as plt

words_in_question = [len(doc) for doc in docs]
words_in_question = np.array(words_in_question)
words_in_sincere_questions = np.take(words_in_question, sincere_indices)
words_in_insincere_questions = np.take(words_in_question, insincere_indices)

sns.set(style="white", palette="muted", color_codes=True)
sns.distplot(words_in_sincere_questions, color="b", label="sincere")
sns.distplot(words_in_insincere_questions, color="r", label="insincere")

plt.legend()
plt.show()

dist_file = "words_distribution_" + str(q_fraction) + ".png"
plt.savefig(dist_file)
# print(np.bincount(words_in_sincere_questions))

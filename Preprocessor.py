from collections import defaultdict
import numpy as np
import operator
import pandas as pd
import re
from tqdm import tqdm

# %% Define the preprocessor
class Preprocessor:
    def __init__(self, embeddings: dict):
        self.embeddings = embeddings

    def build_tf_dict(self, sentences: list):
        """
        Build a simple TF (term frequency) dictionary for all words in the provided sentences.
        """
        tf_dict = defaultdict(int)
        for sentence in sentences:
            for word in sentence:
                tf_dict[word] += 1
        return tf_dict

    def check_coverage(self, tf_dictionary: dict):
        """
        Build a simple list of words that are not embedded. Can be used down the stream to preprocess them to something
        known.
        """
        in_vocabulary = defaultdict(int)
        out_of_vocabulary = defaultdict(int)
        in_count = 0
        out_count = 0

        for word in tqdm(tf_dictionary):
            if word in self.embeddings:
                in_vocabulary[word] = embeddings[word]
                in_count += tf_dictionary[word]
            else:
                out_of_vocabulary[word] = tf_dictionary[word]
                out_count += tf_dictionary[word]

        percent_tf = len(in_vocabulary) / len(tf_dictionary)
        percent_all = in_count / (in_count + out_count)
        print('Found embeddings for {:.2%} of vocabulary and {:.2%} of all text'.format(percent_tf, percent_all))

        return sorted(out_of_vocabulary.items(), key=operator.itemgetter(1))[::-1]

    def clean_punctuation(self, text: list):
        result = text

        #TODO Wrong - don't replace the `result` with a single `text`
        for punct in "/-'":
            result = result.replace(punct, ' ')
        for punct in '&':
            result = result.replace(punct, f' {punct} ')
        for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
            result = result.replace(punct, '')

        return result

    def clean_digits(self, text: list):
        result = text

        return result

    def clean_misspelling(self, text: list):
        mispell_dict = {'colour':'color',
                'centre':'center',
                'didnt':'did not',
                'doesnt':'does not',
                'isnt':'is not',
                'shouldnt':'should not',
                'favourite':'favorite',
                'travelling':'traveling',
                'counselling':'counseling',
                'theatre':'theater',
                'cancelled':'canceled',
                'labour':'labor',
                'organisation':'organization',
                'wwii':'world war 2',
                'citicise':'criticize',
                'instagram': 'social medium',
                'whatsapp': 'social medium',
                'snapchat': 'social medium'}

        def _get_mispell(mispell_dict):
            mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
            return mispell_dict, mispell_re

        mispellings, mispellings_re = _get_mispell(mispell_dict)

        def replace(match):
            return mispellings[match.group(0)]

        return mispellings_re.sub(replace, text)

    def apply_cleaning_function(self, fn, texts: list, description = ""):
        result = [fn(text) for text in tqdm(texts)]
        sentences = [text.split() for text in result]
        tf_dict = self.build_tf_dict(sentences)
        oov = self.check_coverage(tf_dict)
        print(oov[:10])

        return result

    def preprocess_for_embeddings_coverage(self, texts: list):
        result = texts

        sentences = [text.split() for text in result]
        tf_dict = self.build_tf_dict(sentences)
        oov = self.check_coverage(tf_dict)

        result = self.apply_cleaning_function(lambda x: self.clean_punctuation(x), result, "Cleaning punctuation...")
        result = self.apply_cleaning_function(lambda x: self.clean_digits(x), result, "Cleaning numbers...")
        result = self.apply_cleaning_function(lambda x: self.clean_misspelling(x), result, "Cleaning misspelled words...")

        return result


# %% Load the data
df_train = pd.read_csv("input/train.csv")

# %% Load the embeddings
def load_embeddings(file):
    embeddings = {}
    with open(file, encoding="utf8", errors='ignore') as f:
        def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
        embeddings = dict(get_coefs(*line.split(" ")) for (i, line) in enumerate(tqdm(f)))

    print('Found %s word vectors.' % len(embeddings))
    return embeddings
embeddings = load_embeddings("input/embeddings/glove.840B.300d/glove.840B.300d.txt")

# %% Preprocess
preprocessor = Preprocessor(embeddings)
questions = df_train["question_text"].values
print(questions[:3])
questions = preprocessor.preprocess_for_embeddings_coverage(questions)
print(questions[:3])

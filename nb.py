from __future__ import division
import re
import string
import numpy as np

def remove_punctuation(s):
    """See: http://stackoverflow.com/a/266162
    """
    exclude = set(string.punctuation)
    return ''.join(ch for ch in s if ch not in exclude)

def tokenize(text):
    text = remove_punctuation(text)
    text = text.lower()
    return re.split("\W+", text)

def count_words(words):
    wc = {}
    for word in words:
        wc[word] = wc.get(word, 0.0) + 1.0
    return wc

import glob

# setup some structures to store our data
vocab = {}
word_counts = {
    "crypto": {},
    "dino": {}
}
prior_count = {
    "crypto": 0.,
    "dino": 0.
}
docs = []

for f in glob.glob("./sample-data/*/*.txt"):
    f = f.strip()
    if not f.endswith(".txt"):
        # skip non .txt files
        continue
    elif "cryptid" in f:
        category = "crypto"
    else:
        category = "dino"
    docs.append((category, f))
    # ok time to start counting stuff...
    prior_count[category] += 1
    text = open(f).read()
    words = tokenize(text)
    counts = count_words(words)
    for word, count in list(counts.items()):
        # if we haven't seen a word yet, let's add it to our dictionaries with a count of 0
        if word not in vocab:
            vocab[word] = 0.0  # use 0.0 here so Python does "correct" math
        if word not in word_counts[category]:
            word_counts[category][word] = 0.0
        vocab[word] += count
        word_counts[category][word] += count


new_doc = open("examples/Allosaurus.txt").read()
#new_doc = open("examples/Python.txt").read()
#new_doc = open("examples/Yeti.txt").read()

words = tokenize(new_doc)
counts = count_words(words)


# P(dino)
prior_dino = (prior_count["dino"] / (prior_count["dino"] + prior_count["crypto"]))
# P(crypto)
prior_crypto = (prior_count["crypto"] / (prior_count["dino"] + prior_count["crypto"]))

print("Prior(dino)  :", prior_dino)
print("Prior(crypto):", prior_crypto)

log_score_dino = np.log(prior_dino)
log_score_crypto = np.log(prior_crypto)

for w, cnt in list(counts.items()):
    # skip words that we haven't seen before, or words less than 3 letters long
    if w not in vocab or len(w) <= 3:
        continue

    # number of times this word is in all dino articles
    # over count of all words in all dino articles
    p_w_given_dino = (word_counts["dino"].get(w, 0.0) + 1.0) / \
            sum(np.array(word_counts["dino"].values()) + 1.0)

    p_w_given_crypto = (word_counts["crypto"].get(w, 0.0) + 1.0) / \
            sum(np.array(word_counts["crypto"].values()) + 1.0)

    log_score_dino += (cnt * np.log(p_w_given_dino))
    log_score_crypto += (cnt * np.log(p_w_given_crypto))

print("LogScore(dino)  :", log_score_dino)
print("LogScore(crypto):", log_score_crypto)

c1 = log_score_dino - log_score_crypto
c2 = log_score_crypto - log_score_dino

print("P(dino)  :", 1/(np.exp(c2) + 1))
print("P(crypto):", 1/(np.exp(c1) + 1))

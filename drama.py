# Helper functions for analysis.ipynb.
# Downloading, reading files, counting words, splitting the data.
# The models themselves are in the notebook.

import os
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
import requests

FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(FOLDER, 'data')
TEXTS = os.path.join(DATA, 'texts')
RESULTS = os.path.join(FOLDER, 'results')

GENRES = ['comedy', 'tragedy']

# which genre label to use: 'dracor' = their normalizedGenre,
# 'german' = my own reading of the German subtitle
LABEL_SOURCE = 'dracor'

META_URL = 'https://dracor.org/api/v1/corpora/ger/metadata/csv'
TEXT_URL = 'https://dracor.org/api/v1/corpora/ger/plays/{play}/spoken-text'
PEOPLE_URL = 'https://dracor.org/api/v1/corpora/ger/plays/{play}/characters'
HEADERS = {'User-Agent': 'uni-trier-hausarbeit'}

# one session for every download: it keeps the connection open instead of
# opening a new one for each of several hundred requests
SESSION = requests.Session()
SESSION.headers.update(HEADERS)


# ---------------------------------------------------------------- labels

COMEDY_WORDS = ['lustspiel', 'komödie', 'komodie', 'komoedie', 'posse', 'schwank']
TRAGEDY_WORDS = ['trauerspiel', 'tragödie', 'tragodie', 'tragoedie']


def label_from_subtitle(subtitle):
    text = str(subtitle).lower()

    # 'Tragikomödie' contains 'komödie', so it has to go first
    if 'tragikom' in text or 'tragi-kom' in text:
        return None

    is_comedy = False
    for word in COMEDY_WORDS:
        if word in text:
            is_comedy = True

    is_tragedy = False
    for word in TRAGEDY_WORDS:
        if word in text:
            is_tragedy = True

    if is_comedy and not is_tragedy:
        return 'comedy'
    if is_tragedy and not is_comedy:
        return 'tragedy'
    return None


def label_from_dracor(value):
    value = str(value).strip().lower()
    if value in GENRES:
        return value
    return None


# ---------------------------------------------------------------- downloading

def corpus_version():
    # GerDraCor keeps growing and has no version numbers, so the git commit
    # is the only way to say which state of it these numbers came from
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, 'provenance.txt')

    if not os.path.exists(path):
        url = 'https://api.github.com/repos/dracor-org/gerdracor/commits'
        answer = requests.get(url, params={'per_page': 1}, timeout=30, headers=HEADERS)
        newest = answer.json()[0]
        note = 'gerdracor commit ' + newest['sha'] + '\n'
        note = note + 'committed  ' + newest['commit']['committer']['date'] + '\n'
        note = note + 'downloaded ' + time.strftime('%Y-%m-%d') + '\n'
        open(path, 'w').write(note)

    return open(path).read().strip()


def get_metadata():
    # the big metadata table, one row per play. downloaded once, then cached.
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, 'metadata.csv')

    if not os.path.exists(path):
        answer = requests.get(META_URL, timeout=60, headers=HEADERS)
        answer.raise_for_status()
        open(path, 'wb').write(answer.content)

    return pd.read_csv(path)


def make_catalog(meta):
    # keep only the columns I need and add both genre labels
    catalog = pd.DataFrame()
    catalog['play'] = meta['name']
    catalog['author'] = meta['firstAuthor'].fillna('unknown')
    catalog['year'] = meta['yearNormalized']
    catalog['subtitle'] = meta['subtitle'].fillna('')
    catalog['genre_raw'] = meta['normalizedGenre']
    catalog['dracor'] = catalog['genre_raw'].map(label_from_dracor)
    catalog['german'] = catalog['subtitle'].map(label_from_subtitle)

    catalog.to_csv(os.path.join(DATA, 'catalog.csv'), index=False, encoding='utf-8')
    return catalog


def download_texts(catalog):
    # spoken text for every play in the catalog you pass in, one file each.
    # already downloaded files are skipped, so this is safe to re-run.
    os.makedirs(TEXTS, exist_ok=True)
    downloaded = 0
    already_had = 0
    failed = 0

    for play in catalog['play']:
        path = os.path.join(TEXTS, str(play) + '.txt')
        if os.path.exists(path):
            already_had = already_had + 1
            continue

        answer = SESSION.get(TEXT_URL.format(play=play), timeout=60)
        if answer.status_code != 200:
            failed = failed + 1
            continue

        answer.encoding = 'utf-8'
        open(path, 'w', encoding='utf-8').write(answer.text)
        downloaded = downloaded + 1

        if downloaded % 50 == 0:
            print('  downloaded', downloaded)

    return {'downloaded': downloaded, 'already had': already_had, 'failed': failed}


def read_texts(catalog):
    # add the downloaded text as a column. rows with no file are dropped.
    rows = []
    texts = []

    for i in catalog.index:
        play = catalog.loc[i, 'play']
        path = os.path.join(TEXTS, str(play) + '.txt')
        if not os.path.exists(path):
            continue
        rows.append(i)
        texts.append(open(path, encoding='utf-8').read())

    result = catalog.loc[rows].copy()
    result['text'] = texts
    return result.reset_index(drop=True)


def get_characters(catalog):
    # the cast list of every play: who appears in it.
    # saved every 50 plays, so stopping this and starting again picks up
    # where it left off instead of throwing the whole download away.
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, 'characters.csv')

    rows = []
    already_have = set()
    if os.path.exists(path):
        old = pd.read_csv(path)
        rows = old.to_dict('records')
        already_have = set(old['play'])

    todo = []
    for play in catalog['play']:
        if play not in already_have:
            todo.append(play)

    print('cast lists still to fetch:', len(todo))
    failed = 0

    for i in range(len(todo)):
        play = todo[i]
        answer = SESSION.get(PEOPLE_URL.format(play=play), timeout=60)

        if answer.status_code != 200:
            failed = failed + 1
            if failed == 1:
                print('failed:', PEOPLE_URL.format(play=play), answer.status_code)
            continue

        for person in answer.json():
            rows.append({'play': play, 'character': person.get('name', '')})

        if (i + 1) % 50 == 0:
            pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8')
            print('  ', i + 1, 'of', len(todo))

    if failed > 0:
        print('failed:', failed)

    table = pd.DataFrame(rows)
    table.to_csv(path, index=False, encoding='utf-8')
    return table


# ---------------------------------------------------------------- words

WORD = re.compile(r'[a-zäöüß]+')


def tokenize(text):
    # lowercase, letters only. no numbers, no punctuation.
    return WORD.findall(text.lower())


def count_words(texts, vocab_size=3000, min_plays=5):
    # build the word list, then count. one row per play, one column per word.
    all_tokens = []
    for text in texts:
        all_tokens.append(tokenize(text))

    plays_per_word = Counter()
    for tokens in all_tokens:
        plays_per_word.update(set(tokens))

    words = []
    for word, n in plays_per_word.most_common():
        if n >= min_plays and len(words) < vocab_size:
            words.append(word)

    return count_with_words(texts, words), words


def count_with_words(texts, words):
    # count using a word list that already exists, so that plays scored later
    # line up with the plays the model was trained on
    position = {}
    for i in range(len(words)):
        position[words[i]] = i

    counts = np.zeros((len(texts), len(words)))
    for row in range(len(texts)):
        for word, n in Counter(tokenize(texts[row])).items():
            if word in position:
                counts[row, position[word]] = n
    return counts


def to_shares(counts):
    # counts -> share of that play's words, so long plays do not dominate
    total = counts.sum(axis=1, keepdims=True)
    total[total == 0] = 1
    return counts / total


# ---------------------------------------------------------------- splitting

def majority_share(labels):
    # what you score by always guessing the commoner class
    counts = pd.Series(labels).value_counts()
    return counts.iloc[0] / len(labels)


def split_by_author(plays, test_share=0.3, seed=0):
    # no author on both sides, so the model faces playwrights it never read
    rng = np.random.RandomState(seed)
    authors = sorted(plays['author'].unique())
    rng.shuffle(authors)

    held_out = set()
    n = 0
    for author in authors:
        if n >= test_share * len(plays):
            break
        held_out.add(author)
        n = n + (plays['author'] == author).sum()

    test = plays['author'].isin(held_out).values
    return ~test, test


def split_randomly(plays, test_share=0.3, seed=0):
    # the careless version, for comparison. authors leak across this split.
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(plays))
    how_many = int(test_share * len(plays))

    test = np.zeros(len(plays), bool)
    test[order[:how_many]] = True
    return ~test, test


def folds_by_author(plays, k=5, seed=0):
    # k author-disjoint folds, so every play gets one prediction from a model
    # that never saw it
    rng = np.random.RandomState(seed)
    authors = sorted(plays['author'].unique())
    rng.shuffle(authors)

    fold_of = {}
    for i in range(len(authors)):
        fold_of[authors[i]] = i % k

    which = plays['author'].map(fold_of).values
    for i in range(k):
        yield which != i, which == i


def purity(clusters, labels):
    # share of plays whose cluster's most common label is also their own
    correct = 0
    for c in np.unique(clusters):
        inside = labels[clusters == c]
        biggest = np.unique(inside, return_counts=True)[1].max()
        correct = correct + biggest
    return correct / len(labels)


def save(figure, name):
    os.makedirs(RESULTS, exist_ok=True)
    figure.savefig(os.path.join(RESULTS, name), dpi=150, bbox_inches='tight')

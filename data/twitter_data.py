import bz2
import json
import spacy
from profanity_filter import ProfanityFilter
import numpy as np

nlp = spacy.load('en')
profanity_filter = ProfanityFilter(nlps={'en': nlp})
nlp.add_pipe(profanity_filter.spacy_component, last=True)


def construct_dict(files):
    lines = []
    search_dict = {}
    for filename in files:
        with bz2.open(filename, "rt") as bzinput:
            for i, line in enumerate(bzinput):
                # if i == 100: break
                tweets = json.loads(line)
                if "text" not in tweets:
                    continue
                if tweets["lang"] != "en":
                    continue
                if tweets["retweeted"]:
                    continue
                lines.append(tweets)

                words = line["text"].split()
                for word in words:
                    w = word.lower()
                    if w in search_dict:
                        search_dict[w].append(line["id_str"])
                    else:
                        search_dict[w] = [line["id_str"]]
    return lines, search_dict

def get_tweets(tgt, lines):
  for line in lines:
    if "text" not in line:
      continue
    if line["lang"] != "en":
      continue
    else:
      if line["id_str"] in tgt:
        yield line["text"]

def get_samples(new_term, dict, lines):

    if new_term in dict:
        ids = dict[new_term]
    else:
        return None

    tweets = get_tweets(ids, lines)

    filtered = []
    for tweet in tweets:
        doc = nlp(tweet)
        if not doc._.is_profane:
            filtered.append(tweets)

    return filtered


def sample_defs(def_list, new_term, num_samples= 4):

    samples = []

    while len(samples) <= num_samples:
        sample = np.random.sample(def_list)
        if sample["word"] == new_term:
            continue
        definition = sample["definition"]

        if not nlp(definition)._.is_profane:
            samples.append(sample)

    return samples




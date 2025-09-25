import os

import numpy as np

from tqdm import tqdm
from collections import defaultdict

from .utils import store


class Embeddings(object):
    """Class to load, edit and store word embeddings.

    Attr:
        X: embedding matrix
        W: list of words
        Wset: set of words
    """

    def __init__(self, log):
        """Initalize the wrapper

        Args:
            log: a logger object
        """
        self.log = log

    def load_from_gensim(self, model):
        self.path = None
        self.W = model.index_to_key
        self.Wset = set(self.W)
        self.X = model.vectors
        
    def load(self, path, load_first_n=None, header=True):
        """Load word embeddings in word2vec format from a txt file.

        Args:
            path: path to the embedding file
            load_first_n: int; how many lines to load
            header: bool; whether the embedding file contains a header line
        """
        self.path = path
        self.log.info("loading embeddings: {}".format(self.path))

        fin = open(self.path, 'r')

        if header:
            n, d = map(int, fin.readline().split())
        else:
            n, d = None, None

        data = {}
        count = 0
        for line in tqdm(fin):
            count += 1
            if load_first_n is not None and count > load_first_n:
                break
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = list(map(float, tokens[1:]))

        self.W = list(data.keys())
        self.Wset = set(self.W)
        self.X = np.vstack(tuple([data[x] for x in self.W]))

        self.log.info("loaded {} / {} vectors with dimension {}.".format(len(self.W), n, self.X.shape[1]))

    def normalize(self):
        """Normalize the embeddings with l2 norm
        """
        self.X = (self.X.transpose() / np.linalg.norm(self.X, axis=1)).transpose()

    def filter(self, relevant):
        """Filter the embeddings to contain only words from "relevant".

        Args:
            relevant: iterable of words which should be kept
        """
        relevant = set(relevant)
        choose = []
        for word in self.W:
            if word in relevant:
                choose.append(True)
            else:
                choose.append(False)
        self.W = list(np.array(self.W)[choose])
        self.Wset = set(self.W)
        self.X = self.X[choose]

        self.log.info("filtered for {} / {} words.".format(len(relevant), len(self.W)))

    def store(self, fname):
        """Store the embedding space

        Args:
            fname: path to the file
        """
        outfile = store(fname)
        n, dim = self.X.shape
        outfile.write("{} {}\n".format(n, dim))
        for i in range(n):
            outfile.write(self.W[i])
            for k in range(dim):
                outfile.write(" {}".format(self.X[i, k]))
            outfile.write("\n")
        outfile.close()


class Lexicon(object):
    """Class to load, edit and store a lexicon.

    Attr:
        L: dictionary with different versions of the lexicon.
    """

    def __init__(self, log):
        """Initalize the lexicon

        Args:
            log: a logger object
        """
        self.log = log
        self.L = {"countable": [],
                  "ranked": [],
                  "continuous": []}

    def filter_words(self, relevant):
        """Filter the lexicon to contain only words from "relevant".

        Args:
            relevant: iterable of words which should be kept
        """
        relevant = set(relevant)
        for version in self.L:
            tmp = [(k, v) for k, v in self.L[version] if k in relevant]
            self.log.info("Filtering lexicon: {} / {} remaining.".format(len(tmp), len(self.L[version])))
            self.L[version] = tmp

    def load_binary(self, pos_words, neg_words, version):
        self.path = None
        self.L[version] = [(word, 1) for word in pos_words] + [(word, -1) for word in neg_words]
        self.version = version

    def load(self, path, version):
        """Load a lexicon from a file.

        Args:
            path: input path; one line looks like "word\sscore\n"
            version: whether the lexicon is countable, continuous or a ranking; countable has binary (integer) values, continuous float values and ranking reflects a ranking, but the actual values are irrelevant.
        """
        self.path = path
        infile = open(self.path, 'r')
        lexicon = []
        count = 0
        for i, line in enumerate(infile):
            count += 1
            line = line.replace("\n", "")
            try:
                score = line.split()[-1]
                word = line[:-len(score)].strip()
                score = float(score)
                lexicon.append((word, score))
            except:
                self.log.warning("Unexpected format in line {} from {}".format(i, self.path))
        self.log.info("loaded {} / {} lexicon entries.".format(len(lexicon), count))
        self.L[version] = lexicon
        self.version = version

    def remove_inconsistencies(self, remove_all=False):
        """Remove potential inconsistencies from the lexicon.

        Args:
            remove_all: whether to remove all instances of the inconcistency or keep one instance (the first one).
        """
        if remove_all:
            values = defaultdict(list)
            for k, v in self.L[self.version]:
                values[k].append(v)
            inconsistencies = set([k for k, v in values.items() if len(set(v)) > 1])
            self.log.info("Removed {} inconsistencies.".format(len(inconsistencies)))
            self.L[self.version] = [(k, v) for k, v in self.L[self.version] if k not in inconsistencies]
        else:
            seen = {}
            for k, v in self.L[self.version]:
                if k not in seen:
                    seen[k] = v
            self.log.info("Removed {} inconsistencies.".format(len(self.L[self.version]) - len(seen)))
            self.L[self.version] = list(seen.items())

    def binarise(self, mymap=None, neg=None, pos=None):
        """Get a binary version of the lexicon and store it in "countable"

        Args:
            mymap: map from integers to binary values
            neg: interval (e.g. [-float('inf'), 0]) which continuous scores are considered as "-1"; if None us identidy map.
            pos: same as neg for "1"; if None use median as threshold.
        """
        if self.version == 'countable':
            if mymap is None:
                mymap = {1: 1, -1: -1}
            # filter relevant words
            self.L["countable"] = [(k, v) for k, v in self.L["countable"] if v in mymap]
            self.L["countable"] = [(k, int(mymap[v])) for k, v in self.L["countable"]]
        elif self.version == 'continuous':
            if neg is None or pos is None:
                # use median
                median = np.median([v for k, v in self.L['continuous']])
                neg = [-float('inf'), median]
                pos = [median, float('inf')]
            relevant = [k for k, v in self.L["continuous"] if (neg[0] <= v < neg[1]) or (pos[0] <= v < pos[1])]
            self.filter_words(relevant)
            tmp = []
            for k, v in self.L['continuous']:
                if neg[0] <= v <= neg[1]:
                    tmp.append((k, -1))
                elif pos[0] <= float(v) <= pos[1]:
                    tmp.append((k, 1))
            self.L['countable'] = tmp

    def compute_ranks(self):
        """Get a ranked version of the lexicon and store it in "ranked"
        """
        # check number of ties
        n_ties = len(self.L[self.version]) - len(set([v for k, v in self.L[self.version]]))
        self.log.info("Computing ranks. No. of ties: {} / {}".format(n_ties, len(self.L[self.version])))
        tmp = sorted(self.L[self.version], key=lambda x: x[1])
        self.L['ranked'] = [(k, i) for i, (k, _) in enumerate(tmp)]

    def store(self, fname, version=None):
        """Store the lexicon.

        Args:
            fname: path where to store
            version: if given, just store the specific version of the lexicon.
        """
        if version is None:
            for version in self.L:
                outfile = store(fname + "_" + version + ".txt")
                for k, v in self.L[version]:
                    outfile.write("{} {}\n".format(k, v))
                outfile.close()
        else:
            outfile = store(fname)
            for k, v in self.L[version]:
                outfile.write("{} {}\n".format(k, v))
            outfile.close()

    def normalize(self):
        """Min-Max Normalize the continuous lexicon.
        """
        score_min = min([v for k, v in self.L['continuous']])
        score_max = max([v for k, v in self.L['continuous']])
        self.L['continuous'] = [(k, (v - score_min) / (score_max - score_min))for k, v in self.L['continuous']]

    def invert(self):
        """Invert the continuous lexicon.
        """
        self.L['continuous'] = [(k, -v) for k, v in self.L['continuous']]

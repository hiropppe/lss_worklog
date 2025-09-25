import os

import numpy as np

from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cross_decomposition import CCA
from scipy.linalg import null_space
from scipy import stats
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

from .utils import store


class DensRay(object):
    """Implements the DensRay method.

    There are two basid models: "binary" and "continuous".
    For binary input lexicons both are equivalent.
    See the paper for more details.
    """

    def __init__(self, log, Embeddings, Lexicon):
        """Initialize DensRay

        Args:
            log: logger object
            Embeddings: embedding object
            Lexicon: the lexicon which is used to fit DensRay
        """
        self.log = log
        self.embed = Embeddings
        self.lexic = Lexicon

    def fit(self, weights=None, model='binary', normalize_D=True, normalize_labels=True):
        """Fit DensRay

        Args:
            weights: only for binary model; how to weight the two
                summands; if none: apply dynamic weighting. Example input: [1.0, 1.0]
            model: 'binary' or 'continuous'; which model version of Densray to use
            normalize_D: bool whether to normalize the difference vectors with l2 norm
            normalize_labels: bool whether to normalize the predicted labels.
        """
        if model == 'binary':
            self.prepare_data_binary()
            self.computeA_binary_part1(normalize_D=normalize_D)
            self.computeA_binary_part2(weights=weights)
        elif model == 'continuous':
            self.prepare_data_continuous()
            self.computeA_continuous(
                normalize_D=normalize_D, normalize_labels=normalize_labels)
        else:
            raise NotImplementedError
        self.compute_trafo()

    def prepare_data_binary(self):
        """Data preparation function for the binary model

        It selects the relevant vectors from the embedding space.
        """
        Lrel = [(k, v)
                for k, v in self.lexic.L['countable'] if k in self.embed.Wset]
        values = set([v for k, v in Lrel])
        assert len(values) == 2
        v1, v2 = values
        indexpos = []
        indexneg = []
        for k, v in Lrel:
            if v == v1:
                indexpos.append(self.embed.W.index(k))
            else:
                indexneg.append(self.embed.W.index(k))
        self.Xpos = self.embed.X[indexpos, :]
        self.Xneg = self.embed.X[indexneg, :]
        self.npos = self.Xpos.shape[0]
        self.nneg = self.Xneg.shape[0]

    def prepare_data_continuous(self):
        """Data preparation function for the continuous model

        It selects the relevant vectors from the embedding space.
        """
        if len(self.lexic.L['continuous']) == 0:
            self.log.warning(
                "No continuous labels available, using countable labels instead.")
            self.lexic.L['continuous'] = self.lexic.L['countable']
        self.Wrel = [k for k, v in self.lexic.L['continuous']
                     if k in self.embed.Wset]
        self.scoresrel = np.array(
            [v for k, v in self.lexic.L['continuous'] if k in self.Wrel])
        self.Xrel = self.embed.X[[self.embed.W.index(x) for x in self.Wrel]]

    @staticmethod
    def outer_product_sub_binary(v, M, normD):
        """Helper function to compute the sum of outer products

        While it is not very readable, it is more efficient than
        a brute force implementation.
        """
        D = v.transpose() - M
        if normD:
            norm = np.linalg.norm(D, axis=1)
            D[norm == 0.0] = 0.0
            norm[norm == 0.0] = 1.0
            D = D / norm[:, np.newaxis]
        return D.transpose().dot(D)

    @staticmethod
    def outer_product_sub_continuous(v, M, i, gammas, normD):
        """Helper function to compute the sum of outer products

        While it is not very readable, it is more efficient than
        a brute force implementation.
        """
        D = v.transpose() - M
        gamma = gammas[i] * gammas
        if normD:
            norm = np.linalg.norm(D, axis=1)
            D[norm == 0.0] = 0.0
            norm[norm == 0.0] = 1.0
            D = D / norm[:, np.newaxis]
        D1 = gamma[:, np.newaxis] * D
        return D1.transpose().dot(D)

    def store(self, fname):
        """Stores the transformation in npy format.

        Args:
            fname: path where to store the transformation.
        """
        np.save(fname, self.T)

    def computeA_binary_part1(self, normalize_D=True):
        """First part of computing the matrix A.

        Args:
            normalize_D: bool whether to normalize the difference vectors with l2 norm.

        Todo:
            can be made more efficient (dot product is symmetric and we compute both directions here)
        """
        dim = self.Xpos.shape[1]
        self.A_equal = np.zeros((dim, dim))
        self.A_unequal = np.zeros((dim, dim))
        for ipos in tqdm(range(self.npos), desc="compute matrix part1", leave=False):
            v = self.Xpos[ipos:ipos + 1, :].transpose()
            self.A_equal += self.outer_product_sub_binary(
                v, self.Xpos, normalize_D)
            self.A_unequal += self.outer_product_sub_binary(
                v, self.Xneg, normalize_D)
        for ineg in tqdm(range(self.nneg), desc="compute matrix part2", leave=False):
            v = self.Xneg[ineg:ineg + 1, :].transpose()
            self.A_equal += self.outer_product_sub_binary(
                v, self.Xneg, normalize_D)
            self.A_unequal += self.outer_product_sub_binary(
                v, self.Xpos, normalize_D)

    def computeA_binary_part2(self, weights=None):
        """Second part of computing the matrix A.

        Args:
            weights: only for binary model; how to weight the two
                summands; if none: apply dynamic weighting. Example input: [1.0, 1.0]
        """
        if weights is None:
            weights = [1 / (2 * self.npos * self.nneg), 1 /
                       (self.npos**2 + self.nneg**2)]
        # normalize matrices for numerical reasons
        # note that this does not change the eigenvectors
        n1 = self.A_unequal.max()
        n2 = self.A_equal.max()
        weights = [weights[0] / max(n1, n2), weights[1] / max(n1, n2)]
        self.A = weights[0] * self.A_unequal - weights[1] * self.A_equal

    def computeA_continuous(self, normalize_D=True, normalize_labels=True):
        """Compute the matrix A for the continuous case.

        Args:
            normalize_D: normalize_D: bool whether to normalize the difference vectors with l2 norm.
            normalize_labels: bool whether to normalize the predicted labels.

        Todo:
            can be made more efficient (dot product is symmetric and we compute both directions here)
        """
        dim = self.Xrel.shape[1]
        self.A = np.zeros((dim, dim))
        gammas = self.scoresrel
        if normalize_labels:
            gammas = (gammas - gammas.mean()) / gammas.std()
        for i, w in tqdm(enumerate(self.Wrel), desc="compute matrix", leave=False):
            v = self.Xrel[i:i + 1, :].transpose()
            self.A += self.outer_product_sub_continuous(
                v, self.Xrel, i, gammas, normalize_D)
        self.A = - self.A / self.A.max()

    def compute_trafo(self):
        """Given A, this function computes the actual Transformation.

        It essentially just does an eigenvector decomposition.
        """
        # note that (eigvecs(A) = eigvecs (A'A))
        # when using eigh the are always real
        self.eigvals, self.eigvecs = np.linalg.eigh(self.A)
        # need to sort the eigenvalues
        idx = self.eigvals.argsort()[::-1]
        self.eigvals, self.eigvecs = self.eigvals[idx], self.eigvecs[:, idx]
        self.T = self.eigvecs
        assert np.allclose(self.T.transpose().dot(self.T), np.eye(
            self.T.shape[0])), "self.T not orthonormal."


class Regression(object):
    """Implements a regression based method of obtaining interpretable dimensions.

    Different models from sklearn are available.
    The word "regression" is not really appropriate, as one can also apply SVMs.
    All models: ["logistic", "svm", "linear", "svr", "cca"]
    """

    def __init__(self, log, Embeddings, Lexicon):
        """Initialize the Regression

        Args:
            log: logger object
            Embeddings: embedding object
            Lexicon: the lexicon which is used to fit the model
        """
        self.log = log
        self.embed = Embeddings
        self.lexic = Lexicon

    def prepare_data(self, model, add_random_words=False):
        """Prepare the data (i.e. select vectors and create labels)

        Args:
            model: string; a value in ["logistic", "svm", "linear", "svr", "cca"]
        """
        if model in ["logistic", "svm"]:
            version = 'countable'
        elif model in ["linear", "svr", "cca"]:
            version = self.lexic.version
        else:
            raise ValueError("Model unknown.")
        idxs = []
        ys = []
        words = []
        for k, v in self.lexic.L[version]:
            if k in self.embed.Wset:
                idxs.append(self.embed.W.index(k))
                ys.append(v)
                words.append(k)
        if add_random_words:
            n_add = sum([y == 1 for y in ys])
            idx_to_add = np.random.choice(len(self.embed.Wset), n_add)
            words_to_add = [self.embed.W[x] for x in idx_to_add]
            ys_to_add = [0] * n_add

            idxs.extend(idx_to_add)
            ys.extend(ys_to_add)
            words.extend(words_to_add)

        self.Wrel = words
        self.Xrel = self.embed.X[idxs, :]
        self.Yrel = np.array(ys)
        if model == 'logistic':
            self.Yrel[self.Yrel == -1] = 0
        if model == 'svm':
            self.Yrel[self.Yrel == 0] = -1

    def fit(self, model):
        """Fits the model and creates a (random) orthogonal transformation.

        Args:
            model: string; a value in ["logistic", "svm", "linear", "svr", "cca"]
        """
        if model == 'linear':
            self.mod = LinearRegression()
        elif model == 'logistic':
            self.mod = LogisticRegression(
                penalty='none', class_weight='balanced', solver='saga')
            #self.mod = LogisticRegression()
        elif model == 'svr':
            self.mod = SVR(kernel='linear')
        elif model == 'svm':
            self.mod = SVC(C=1.0, kernel='linear')
        elif model == 'cca':
            self.mod = CCA(n_components=1, scale=True,
                           max_iter=500, tol=1e-06, copy=True)
            self.mod.intercept_ = 0.0
        self.mod.fit(self.Xrel, self.Yrel)
        # now compute T with a random orthogonal basis
        # todo potential bug: what to do with the intercept_?
        w0 = self.mod.coef_  # + self.mod.intercept_
        if len(w0.shape) < 2:
            w0 = w0.reshape(1, -1)
        w0 = w0 / np.linalg.norm(w0)
        Wcompl = null_space(w0)
        self.T = np.hstack((w0.transpose(), Wcompl))
        assert np.allclose(self.T.transpose().dot(self.T), np.eye(
            self.T.shape[0])), "self.T not orthonormal."

    def store(self, fname):
        """Stores the transformation in npy format.

        Args:
            fname: path where to store the transformation.
        """
        np.save(fname, self.T)


class LexIndPredictor(object):
    """Given an interpretable word space and queries, predict lexical scores (e.g. for sentiment.
    """

    def __init__(self, log, embeddings, queries, T):
        """Initialize the predictor.

        Args:
            log: logger object
            embeddings: word embedding object
            queries: list of strings with queries
            T: np.array; linear transformation
        """
        self.log = log
        self.embeds = embeddings
        self.queries = queries
        self.T = T

    def predict(self, method, dim_weights=None):
        """Predict scores for the query words.

        Args:
            method: string; either "first_dimension", "first_n_dimensions"
            dim_weights: only available if method == "first_n_dimensions"; how to weight the scores in the first n dimensions.
        """
        X_trafo = self.embeds.X.dot(self.T)
        self.predictions = []
        for k in self.queries:
            if k not in self.embeds.Wset:
                score = 0.0
            elif method == 'first_dimension':
                score = X_trafo[self.embeds.W.index(k), 0]
            elif method == 'first_n_dimensions':
                n = len(dim_weights)
                score = 0.0
                for i in range(n):
                    score += dim_weights[i] * \
                        X_trafo[self.embeds.W.index(k), i]
            self.predictions.append((k, score))

    def store(self, fname):
        """Stores the predictions in a text file.

        Args:
            fname: path where to store the predictions.
        """
        outfile = store(fname)
        for k, v in self.predictions:
            outfile.write("{} {}\n".format(k, v))
        outfile.close()

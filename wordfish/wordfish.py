"""
WORDFISH: A Scaling Model for Estimating Time-Series Party Positions from Texts

Python implementation of the WORDFISH algorithm originally developed by
Jonathan B. Slapin and Sven-Oliver Proksch (2008).

Author: Sonnet4.1

Reference:
Slapin, J. B., & Proksch, S. O. (2008). A scaling model for estimating
time-series party positions from texts. American Journal of Political Science,
52(3), 705-722.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import svd
import warnings
from typing import Dict, List, Optional, Tuple, Union


class WordfishResults:
    """Container for WORDFISH estimation results."""

    def __init__(self, documents: np.ndarray, words: np.ndarray,
                 estimation_info: Dict, doc_names: List[str],
                 word_names: List[str]):
        self.documents = documents  # omega (positions) and alpha (fixed effects)
        self.words = words  # beta (weights) and psi (fixed effects)
        self.estimation_info = estimation_info
        self.doc_names = doc_names
        self.word_names = word_names

    def __repr__(self):
        return f"WordfishResults(documents={self.documents.shape[0]}, words={self.words.shape[0]})"


class Wordfish:
    """
    WORDFISH scaling model implementation.

    This class implements the WORDFISH algorithm for estimating one-dimensional
    positions from texts using a Poisson model with word and document fixed effects.

    Model: y_ij ~ Poisson(exp(alpha_i + psi_j + omega_i * beta_j))

    Where:
    - alpha_i: document i fixed effect (document 1 has alpha=0)
    - omega_i: document i position on latent dimension
    - psi_j: word j fixed effect
    - beta_j: word j weight/discrimination parameter
    """

    def __init__(self, tol: float = 1e-7, sigma: float = 3.0,
                 max_iter: int = 1000, verbose: bool = True):
        """
        Initialize WORDFISH estimator.

        Parameters
        ----------
        tol : float, default=1e-7
            Convergence tolerance criterion
        sigma : float, default=3.0
            Variance parameter to constrain beta (prior standard deviation)
        max_iter : int, default=1000
            Maximum number of iterations
        verbose : bool, default=True
            Whether to print progress information
        """
        self.tol = tol
        self.sigma = sigma
        self.max_iter = max_iter
        self.verbose = verbose

    def fit(self, word_doc_matrix: Union[np.ndarray, pd.DataFrame],
            dir_docs: Tuple[int, int] = (0, 1),
            doc_names: Optional[List[str]] = None,
            word_names: Optional[List[str]] = None) -> WordfishResults:
        """
        Fit WORDFISH model to word-document matrix.

        Parameters
        ----------
        word_doc_matrix : array-like, shape (n_documents, n_words)
            Term-document matrix where rows are documents and columns are words
        dir_docs : tuple of int, default=(0, 1)
            Two document indices for global identification (ensures consistent sign)
        doc_names : list of str, optional
            Names for documents (rows)
        word_names : list of str, optional
            Names for words (columns)

        Returns
        -------
        WordfishResults
            Fitted model results containing document positions, word parameters, etc.
        """
        # Convert input to numpy array
        if isinstance(word_doc_matrix, pd.DataFrame):
            if doc_names is None:
                doc_names = word_doc_matrix.index.tolist()
            if word_names is None:
                word_names = word_doc_matrix.columns.tolist()
            dta = word_doc_matrix.values
        else:
            dta = np.array(word_doc_matrix)

        # Set default names if not provided
        if doc_names is None:
            doc_names = [f"doc_{i}" for i in range(dta.shape[0])]
        if word_names is None:
            word_names = [f"word_{i}" for i in range(dta.shape[1])]

        n_docs, n_words = dta.shape

        if self.verbose:
            print("=" * 40)
            print("WORDFISH (Python Implementation)")
            print("=" * 40)
            print(f"Number of unique words: {n_words}")
            print(f"Number of documents: {n_docs}")
            print(f"Tolerance criterion: {self.tol}")
            print(f"Identification: Omegas identified with mean 0, st.dev. 1")
            print("=" * 40)

        # Get starting values
        if self.verbose:
            print("Performing mean 0 sd 1 starting value calc")

        params = self._get_starting_values(dta)

        # Run EM algorithm
        if self.verbose:
            print("Performing mean 0 sd 1 EM algorithm")

        params = self._em_algorithm(dta, params, dir_docs)

        if self.verbose:
            print("=" * 40)
            print("WORDFISH ML Estimation finished.")
            print("=" * 40)

        # Prepare output
        output_documents = np.column_stack([params['omega'], params['alpha']])
        output_words = np.column_stack([params['beta'], params['psi']])

        estimation_info = {
            'n_words': n_words,
            'n_documents': n_docs,
            'iterations': params['iter'],
            'log_likelihood': params['maxllik'][-1] if params['maxllik'] else None,
            'convergence_criterion': self.tol,
            'final_diff': params.get('diffparam_last', None)
        }

        return WordfishResults(
            documents=output_documents,
            words=output_words,
            estimation_info=estimation_info,
            doc_names=doc_names,
            word_names=word_names
        )

    def _get_starting_values(self, dta: np.ndarray) -> Dict:
        """Calculate starting values using SVD decomposition."""
        n_docs, n_words = dta.shape

        # Starting values for psi (word fixed effects)
        psi = np.log(np.mean(dta, axis=0) + 1e-8)  # Add small constant to avoid log(0)

        # Starting values for alpha (document fixed effects)
        doc_means = np.mean(dta, axis=1)
        alpha = np.log(doc_means / doc_means[0] + 1e-8)

        # Starting values for beta and omega using SVD
        # Create residuals after removing fixed effects
        ystar = np.log(dta + 0.1)  # Add constant like in R version
        for i in range(n_docs):
            for j in range(n_words):
                ystar[i, j] -= alpha[i] + psi[j]

        # SVD decomposition
        U, s, Vt = svd(ystar, full_matrices=False)

        # Extract first component
        beta = Vt[0, :] * s[0]  # First right singular vector * first singular value
        omega1 = U[:, 0] - U[0, 0]  # First left singular vector, normalized

        # Standardize omega to mean 0, sd 1
        omega = omega1 / np.std(omega1) if np.std(omega1) > 0 else omega1
        beta = beta * np.std(omega1) if np.std(omega1) > 0 else beta

        return {
            'alpha': alpha,
            'psi': psi,
            'beta': beta,
            'omega': omega,
            'iter': 0,
            'maxllik': [],
            'diffllik': []
        }

    def _llik_psi_beta(self, params: np.ndarray, y: np.ndarray,
                      omega: float, alpha: float) -> float:
        """Log-likelihood for psi and beta estimation."""
        beta, psi = params
        lambda_param = np.exp(psi + alpha + beta * omega)

        # Poisson log-likelihood with normal prior on beta
        ll = np.sum(-lambda_param + y * np.log(lambda_param + 1e-10))
        prior = -0.5 * (beta**2 / self.sigma**2)

        return -(ll + prior)  # Return negative for minimization

    def _llik_alpha_omega_first(self, omega: float, y: np.ndarray,
                               beta: np.ndarray, psi: np.ndarray) -> float:
        """Log-likelihood for first document omega (alpha=0)."""
        lambda_param = np.exp(psi + beta * omega)
        ll = np.sum(-lambda_param + y * np.log(lambda_param + 1e-10))
        return -ll  # Return negative for minimization

    def _llik_alpha_omega(self, params: np.ndarray, y: np.ndarray,
                         beta: np.ndarray, psi: np.ndarray) -> float:
        """Log-likelihood for omega and alpha estimation."""
        omega, alpha = params
        lambda_param = np.exp(psi + alpha + beta * omega)
        ll = np.sum(-lambda_param + y * np.log(lambda_param + 1e-10))
        return -ll  # Return negative for minimization

    def _em_algorithm(self, dta: np.ndarray, params: Dict,
                     dir_docs: Tuple[int, int]) -> Dict:
        """Run EM algorithm until convergence."""
        n_docs, n_words = dta.shape
        iteration = 1
        diffllik = 500.0

        # Storage for tracking convergence
        params['maxllik'] = []
        params['diffllik'] = []

        while diffllik > self.tol and iteration <= self.max_iter:
            if self.verbose:
                print(f"Iteration {iteration}")
                print("\tUpdating alpha and omega..")

            # Store previous values
            omega_prev = params['omega'].copy()

            # UPDATE OMEGA AND ALPHA

            # First document: estimate omega (alpha=0)
            res = minimize(
                self._llik_alpha_omega_first,
                x0=params['omega'][0],
                args=(dta[0, :], params['beta'], params['psi']),
                method='BFGS'
            )
            if not res.success:
                warnings.warn("Optimization failed to converge for first document omega")
            params['omega'][0] = res.x
            params['alpha'][0] = 0.0

            # Other documents: estimate both omega and alpha
            for i in range(1, n_docs):
                res = minimize(
                    self._llik_alpha_omega,
                    x0=[params['omega'][i], params['alpha'][i]],
                    args=(dta[i, :], params['beta'], params['psi']),
                    method='BFGS'
                )
                if not res.success:
                    warnings.warn(f"Optimization failed to converge for document {i}")
                params['omega'][i] = res.x[0]
                params['alpha'][i] = res.x[1]

            # Z-score transformation for identification
            omega_bar = np.mean(params['omega'])
            omega_sd = np.std(params['omega'])
            if omega_sd > 0:
                b1 = params['beta'].copy()
                params['beta'] = params['beta'] * omega_sd
                params['omega'] = (params['omega'] - omega_bar) / omega_sd
                params['psi'] = params['psi'] + b1 * omega_bar

            # Global identification (ensure consistent direction)
            if params['omega'][dir_docs[0]] > params['omega'][dir_docs[1]]:
                params['omega'] = -params['omega']
                params['beta'] = -params['beta']

            if self.verbose:
                print("\tUpdating psi and beta..")

            # UPDATE PSI AND BETA
            for j in range(n_words):
                res = minimize(
                    self._llik_psi_beta,
                    x0=[params['beta'][j], params['psi'][j]],
                    args=(dta[:, j], params['omega'], params['alpha']),
                    method='BFGS'
                )
                if not res.success:
                    warnings.warn(f"Optimization failed to converge for word {j}")
                params['beta'][j] = res.x[0]
                params['psi'][j] = res.x[1]

            # Calculate log-likelihood and check convergence
            total_ll = 0.0
            for i in range(n_docs):
                for j in range(n_words):
                    lambda_ij = np.exp(params['alpha'][i] + params['psi'][j] +
                                     params['omega'][i] * params['beta'][j])
                    if dta[i, j] > 0:
                        total_ll += -lambda_ij + dta[i, j] * np.log(lambda_ij)
                    else:
                        total_ll += -lambda_ij

            params['maxllik'].append(total_ll)

            # Calculate convergence criteria
            diffparam = np.mean(np.abs(params['omega'] - omega_prev))

            if len(params['maxllik']) > 1:
                diff_ll = params['maxllik'][-1] - params['maxllik'][-2]
                diffllik = diff_ll / abs(params['maxllik'][-1]) if params['maxllik'][-1] != 0 else 0
            else:
                diffllik = 500.0

            params['diffllik'].append(diffllik)
            params['diffparam_last'] = diffparam

            if self.verbose:
                print(f"\tConvergence of LL: {diffllik}")

            iteration += 1

        params['iter'] = iteration - 1
        return params


def wordfish(word_doc_matrix: Union[np.ndarray, pd.DataFrame],
            tol: float = 1e-7,
            sigma: float = 3.0,
            dir_docs: Tuple[int, int] = (0, 1),
            verbose: bool = True,
            doc_names: Optional[List[str]] = None,
            word_names: Optional[List[str]] = None) -> WordfishResults:
    """
    Convenient function interface for WORDFISH estimation.

    Parameters
    ----------
    word_doc_matrix : array-like, shape (n_documents, n_words)
        Term-document matrix where rows are documents and columns are words
    tol : float, default=1e-7
        Convergence tolerance criterion
    sigma : float, default=3.0
        Variance parameter to constrain beta
    dir_docs : tuple of int, default=(0, 1)
        Two document indices for global identification
    verbose : bool, default=True
        Whether to print progress information
    doc_names : list of str, optional
        Names for documents
    word_names : list of str, optional
        Names for words

    Returns
    -------
    WordfishResults
        Fitted model results
    """
    model = Wordfish(tol=tol, sigma=sigma, verbose=verbose)
    return model.fit(word_doc_matrix, dir_docs=dir_docs,
                    doc_names=doc_names, word_names=word_names)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Create sample word-document matrix
    n_docs, n_words = 5, 10
    # Simulate data with some structure
    true_positions = np.array([-2, -1, 0, 1, 2])
    true_betas = np.random.normal(0, 1, n_words)

    word_doc_matrix = np.zeros((n_docs, n_words))
    for i in range(n_docs):
        for j in range(n_words):
            lambda_ij = np.exp(0.5 + true_positions[i] * true_betas[j])
            word_doc_matrix[i, j] = np.random.poisson(lambda_ij)

    # Fit WORDFISH model
    print("Running WORDFISH on simulated data...")
    results = wordfish(word_doc_matrix, verbose=True)

    print("\nDocument positions (omega):")
    for i, (name, pos) in enumerate(zip(results.doc_names, results.documents[:, 0])):
        print(f"{name}: {pos:.3f} (true: {true_positions[i]:.3f})")
import numpy as np
from scipy.stats import rv_histogram

class logLikelihood():
    def __init__(self):
        """
        mus: values of mu to test
        rescale: correction factor to report significance for 300/fb of data
        rescale = n_test_events / (sigma(jjWpmWpm) * Br(W->lnu)^2 * (300/fb)) = 4.67
        """
        self.mus = np.linspace(0.0, 0.12, 200)
        self.rescale = 32000 / (213.3 * 0.3272**2 * 300)

    def hist_dist(self, data):
        """
        generate a distribution given by a histogram
        """
        hist = np.histogram(data,
                            bins=50,
                            density=True)
        return rv_histogram(hist)

    def mLL_mu(self, mu, pdf_1, pdf_0):
        """
        compute the log-likelihood for a single value of mu
        since we will be interested in the difference of LL from its maximum, only include finite contributions
        """
        mLL = -np.log(mu * pdf_1 + (1 - mu) * pdf_0)
        return np.sum(mLL[np.isfinite(mLL)])

    def compute_log_likelihood(self, clf, X, y):
        """
        given a classifier and labeled data, compute the associated log-likelihood as a function of mu
        """
        probas_all = clf.predict_proba(X)
        probas_1 = clf.predict_proba(X[y == 1])
        probas_0 = clf.predict_proba(X[y == 0])

        # sometimes both p and 1-p are reported
        if probas_all.shape[1] == 2:
            probas_all = probas_all.T[1]
            probas_1 = probas_1.T[1]
            probas_0 = probas_0.T[1]

        hd_1 = self.hist_dist(probas_1)
        hd_0 = self.hist_dist(probas_0)
        pdf_1 = hd_1.pdf(probas_all)
        pdf_0 = hd_0.pdf(probas_all)

        return np.array([self.mLL_mu(mu, pdf_1, pdf_0) for mu in self.mus])

    def compute_sigma(self, m_likelihood, threshold=0.5):
        """
        estimate the number of standard deviations from zero this measurement yields
        """
        arg_cen = np.argmin(m_likelihood)
        central = self.mus[arg_cen]

        delta_likelihood = m_likelihood - np.min(m_likelihood)
        arg_std = np.argmin((delta_likelihood[:arg_cen] - threshold)**2)
        stdev = central - self.mus[arg_std]

        return central / stdev

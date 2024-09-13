import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit
import pandas as pd
import traceback
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
    )

class Curves:
    """
    A class containing various static methods to model different mathematical curves.
    """
    @staticmethod
    def linear(x,a,b):
        x = x.astype(float)
        return a*x*b

    @staticmethod
    def expon(x,a,b):
        x = x.astype(float)
        return a*np.power(x,b)

    @staticmethod
    def expon_neg(x,a,b):
        x = x.astype(float)
        return a*pow(x,-b)

    @staticmethod
    def euler(x, a, b):
        x = x.astype(float)
        return a * np.exp(b * x)

    @staticmethod
    def power(x,a,b):
        x = x.astype(float)
        return a*pow(b,x)

    @staticmethod
    def power_neg(x,a,b):
        x = x.astype(float)
        return a*pow(b,(-x))

    @staticmethod
    def logar(x, a, b):
        x = x.astype(float)
        return a + b * np.log(x)

    @staticmethod
    def cubic(x, a, b, c, d):
        x = x.astype(float)
        return a*x**3 + b*x**2 + c*x + d

    @staticmethod
    def sigmoid(x,a,b):
        x = x.astype(float)
        return 1/(1+np.exp(a*x+b))

    @staticmethod
    def quad(x, a, b, c):
        x = x.astype(float)
        return a*x**2 + b*x + c

    @staticmethod
    def four(x, a, b, c, d, e):
        x = x.astype(float)
        return a*x**4 + b*x**3 + c*x**2 + d*x + e

    @staticmethod
    def zipf(x, a, b):
        return 1 / (x + a)**b


class DistributionFitingTools:
    """
    A class containing tools for fitting distributions and models to data.
    """
    def __init__(self) -> None:
        self.curves = Curves()

    def fit_distribution(self, data, distribution):
        """
        Fits a given distribution to the data

        Parameters:
        ----------
        data : numpy.ndarray
            The data to which the distribution is fitted.
        distribution : scipy.stats.rv_continuous
            The distribution to fit to the data.

        Returns:
        -------
        float
        """
        params = distribution.fit(data)
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]
        pdf_values = distribution.pdf(data, loc=loc, scale=scale, *arg)
        log_likelihood = np.sum(np.log(pdf_values))
        num_params = len(params)
        aic = 2 * len(params) - 2 * log_likelihood
        aicc = aic + (2 * num_params * (num_params + 1)) / (len(data) - num_params - 1)  # AICc correction
        return aicc


    def calculate_akaike_weights(self,aic_values):
        """
        Calculates Akaike weights from AIC values.

        Parameters:
        ----------
        aic_values : list of float
            The list of AIC values for different models.

        Returns:
        -------
        numpy.ndarray
            The Akaike weights corresponding to each AIC value.
        """
        delta_aic = aic_values - np.min(aic_values)
        exp_term = np.exp(-0.5 * delta_aic)
        weights = exp_term / np.sum(exp_term)
        return weights


    def multiple_distributions(self, data):
        """
        Fits multiple distributions to the data and selects
        the best one based on AICc.

        Parameters:
        ----------
        data : numpy.ndarray
            The data to which the distributions are fitted.

        Returns:
        -------
        scipy.stats.rv_continuous
            The best fitting distribution.
        """
        distributions = [
            stats.lognorm,
            stats.expon,
            stats.powerlaw,
            stats.norm,
            stats.pareto
            ]

        aic_values = []

        for distribution in distributions:
            current_aic = self.fit_distribution(data, distribution)
            aic_values.append(current_aic)

        print(aic_values)
        weights = self.calculate_akaike_weights(aic_values)

        best_index = np.nanargmin(aic_values)
        best_distribution = distributions[best_index]

        print(f"Best Distribution: {best_distribution.name}")
        print(f"AICc Weights: {weights}")
        return best_distribution

    def model_choose(self, vals):
        """
        Chooses the best fitting model from a set of predefined curves
        based on AICc.

        Parameters:
        ----------
        vals : pandas.Series
            The data to which the models are fitted.

        Returns:
        -------
        tuple
            The best fit model, its name, parameters, and a DataFrame
            containing model information.
        """
        scores = {}
        parameters = {}
        expon_pred = None
        for c in [
            self.curves.linear,
            self.curves.expon,
            self.curves.expon_neg,
            self.curves.sigmoid
            ]:
            try:
                params, covar = curve_fit(c,vals.index.values,vals.values)
                num_params = len(params)

                y_pred = c(vals.index,*params)
                if c.__name__ == 'expon':
                    expon_pred = y_pred
                residuals = vals.values - y_pred

                aic = len(vals.index) * np.log(
                    np.mean(residuals ** 2)) + 2 * len(params)
                aicc = aic + (
                    2*num_params * (num_params + 1)
                    ) / (len(vals) - num_params - 1)
                scores[c] = aicc
                parameters[c]=params

            except ValueError as e:
                logging.error(e)
                continue

        min_aicc = min(scores,key=scores.get)
        min_aicc = scores[min_aicc]
        waicc = {
            k:v-min_aicc for k,v in scores.items()
            }
        waicc = {
            k:np.exp(-0.5*v) for k,v in waicc.items()
            }
        waicc = {
            k:v/np.sum(list(waicc.values())) for k,v in waicc.items()
            }

        global_params = pd.DataFrame(
            columns=[
                'curve',
                'weight',
                'param1',
                'param2'
                ]
                )
        for (key1, value1), (key2, value2) in zip(
            waicc.items(), parameters.items()
            ):
            if key1 == key2:
                global_params = global_params._append({
                    "curve" : key2.__name__,
                    'weight' : round(value1,25),
                    'param1' : round(value2[0],25),
                    'param2' : round(value2[1],25)
                }, ignore_index = True)

        best_fit = max(waicc, key=waicc.get)
        params, _ = curve_fit(best_fit, vals.index.values, vals.values)

        print(f'Fitting results: \n{global_params}')
        print(f'Best fit: {best_fit.__name__}, param1: {params[0]}, param2: {params[1]}')
        return best_fit(vals.index,*params), best_fit.__name__, params, global_params, expon_pred
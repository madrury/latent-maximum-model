from typing import List
import numpy as np
import pandas as pd


class LatentMaximumDataGenerator:
    """Sample a dataset from a data model:

        y = a (s - 𝜇 + ε₁)**2 + 𝜷'·x' + b + ε₂
        𝜇 = 𝜷·x

    In the above, s is viwed as a dynamic variable, and the goal of data
    analysis is to identify the "latent maximum" 𝜇. The latent maximum is
    constructed as a linear combination of observed features x.

    Parameters
    ----------
    n_latent_features: int
      The number of features in the vector x.

    n_additive_features: int
      The number of features in the vector x'.

    maximum_noise_std: int
      The standard deviation of the noise variable ε₁.

    residual_std: int
      The standard deviation of the noise variable ε₂.
    """
    def __init__(
        self,
        n_latent_features=10,
        n_additive_features=10,
        maximum_noise_std=0.0,
        residual_std=10.0
    ):
        self.n_latent_features = n_latent_features
        self.latent_coefs = np.random.normal(size=n_latent_features)
        self.n_additive_features = n_additive_features
        self.additive_coefs = np.random.normal(size=n_additive_features)
        self.a = np.random.uniform(low=-1.0, high=-0.25)
        self.intercept = np.random.uniform(low=-2.0, high=2.0)
        self.maximum_noise_std = maximum_noise_std
        self.residual_std = residual_std

    def generate(self, n: int) -> pd.DataFrame:
        latent_features = np.random.normal(size=(n, self.n_latent_features))
        latent_names = [f"L{i}" for i in range(self.n_latent_features)]
        additive_features = np.random.normal(size=(n, self.n_additive_features))
        additive_names = [f"A{i}" for i in range(self.n_additive_features)]
        s = np.random.normal(size=n, scale=3.0)
        mu = latent_features @ self.latent_coefs
        A = additive_features @ self.additive_coefs
        y = (
            self.a * (s - mu + np.random.normal(scale=self.maximum_noise_std, size=n))**2
            + A
            + self.intercept
            + np.random.normal(scale=self.residual_std, size=n)
        )
        df = pd.DataFrame({
            's': s, 'y': y, 'mu': mu, 'A': A, 'intercept': self.intercept
        })
        return pd.concat([
            df,
            pd.DataFrame(latent_features, columns=latent_names),
            pd.DataFrame(additive_features, columns=additive_names),
        ], axis=1)


class LatentMaximumModel():
    """Fit a latent maximum model of the form:

        ŷ ≈ a (s - 𝜇)**2 + 𝜷'·x'
        𝜇 = 𝜷·x + b

    By minimizing the squared error loss function.
    """
    def __init__(
        self,
        learning_rate=0.0001,
        rtol=0.0001,
        n_iter=25000
    ):
        self.latent_coefs = None
        self.additive_coefs = None
        self.a = -0.5
        self.intercept = 0.0
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.rtol = rtol
        self.losses: List[float] = []

    def fit(self, x: np.array, x_latent: np.array, x_additive: np.array, y: np.array):
        self._initialize(x_latent, x_additive)
        for i in range(self.n_iter):
            self._update(x, x_latent, x_additive, y)
            if i >= 2 and self._has_converged():
                break
        return self

    def predict_mu(self, x_latent: np.array):
        return x_latent @ self.latent_coefs

    def predict(self, x: np.array, x_latent: np.array, x_additive: np.array):
        mu = self.predict_mu(x_latent)
        return (
            self.a * (x - mu)**2
            + x_additive @ self.additive_coefs
            + self.intercept
        )

    def _initialize(self, x_additive, x_latent):
        self.additive_coefs = np.zeros(shape=x_additive.shape[1])
        self.latent_coefs = np.zeros(shape=x_latent.shape[1])

    def _update(self, x: np.array, x_latent: np.array, x_additive: np.array, y: np.array):
        N = len(x)
        mu = self.predict_mu(x_latent)
        yhat = self.predict(x, x_latent, x_additive)
        da = - (2/N) * np.sum((y - yhat) * (x - mu) * (x - mu))
        dadditive = - (2/N) * (y - yhat) @ x_additive
        dintercept = - (2/N) * np.sum(y - yhat)
        dlatent = (4 * self.a / N) * ((y - yhat) * (x - mu)) @ x_latent
        self.a -= self.learning_rate * da
        self.intercept -= self.learning_rate * dintercept
        self.additive_coefs -= self.learning_rate * dadditive
        self.latent_coefs -= self.learning_rate * dlatent
        self.losses.append(LatentMaximumModel._loss(y, yhat))

    @staticmethod
    def _loss(y: np.array, yhat: np.array):
        return (1 / len(y)) * np.sum((y - yhat)**2)

    def _has_converged(self):
        return abs(self.losses[-2] - self.losses[-1]) / self.losses[-2] < self.rtol
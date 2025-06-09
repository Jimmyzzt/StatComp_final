import pints
import numpy as np
import matplotlib.pyplot as plt
import pints.plot

# Define a class for the multi-modal normal log-likelihood
class MultiModalNormalLogLikelihood(pints.LogPDF):
    def __init__(self, means, covariances, weights):
        """
        Initializes the multi-modal normal log-likelihood function.
        :param means: List of mean vectors for each mode.
        :param covariances: List of covariance matrices for each mode.
        :param weights: List of weights for each mode.
        """
        self.means = means
        self.covariances = covariances
        self.weights = weights
        self.n_modes = len(means)
        self.dimensions = len(means[0])

    def n_parameters(self):
        return self.dimensions

    def __call__(self, x):
        """
        Evaluates the log-likelihood at point x.
        :param x: Parameter vector at which to evaluate the log-likelihood.
        """
        log_likelihood = -np.inf
        for i in range(self.n_modes):
            diff = x - self.means[i]
            inv_cov = np.linalg.inv(self.covariances[i])
            det_cov = np.linalg.det(self.covariances[i])
            log_likelihood = np.logaddexp(log_likelihood,
                np.log(self.weights[i]) - 0.5 * np.dot(diff.T, np.dot(inv_cov, diff)) - 0.5 * np.log(det_cov) - 0.5 * self.dimensions * np.log(2 * np.pi))
        return log_likelihood

    def evaluateS1(self, x):
        """
        Evaluates the log-likelihood and its gradient at point x.
        :param x: Parameter vector at which to evaluate the log-likelihood and gradient.
        :return: Tuple (log_likelihood, gradient)
        """
        log_likelihood = -np.inf
        gradient = np.zeros(self.dimensions)
        for i in range(self.n_modes):
            diff = x - self.means[i]
            inv_cov = np.linalg.inv(self.covariances[i])
            det_cov = np.linalg.det(self.covariances[i])
            log_term = np.log(self.weights[i]) - 0.5 * np.dot(diff.T, np.dot(inv_cov, diff)) - 0.5 * np.log(det_cov) - 0.5 * self.dimensions * np.log(2 * np.pi)
            if log_term > log_likelihood:
                log_likelihood = log_term
                gradient = -np.dot(inv_cov, diff)
        return log_likelihood, gradient

# Define the number of dimensions for the multi-modal normal distribution
dimensions = 2  # Change this to test different dimensions

# Define the means, covariances, and weights for the multi-modal normal distribution
means = [
    np.array([0.0, 0.0]),
    np.array([5.0, 5.0]),
    np.array([-5.0, 5.0]),
    np.array([5.0, -5.0]),
    np.array([-5.0, -5.0]),
]
covariances = [np.eye(dimensions) for _ in range(len(means))]
weights = [1.0 / len(means) for _ in range(len(means))]

# Create the log-likelihood function
log_likelihood = MultiModalNormalLogLikelihood(means, covariances, weights)

# Define a uniform prior over the parameters
# Assuming the parameters are within [-10, 10] for each dimension
log_prior = pints.UniformLogPrior(
    [-10] * dimensions,
    [10] * dimensions
)

# Create a posterior log-likelihood (log(likelihood * prior))
log_posterior = pints.LogPosterior(log_likelihood, log_prior)

# Choose starting points for 3 MCMC chains
# Initialize near the mean with some small perturbation
xs = [
    np.random.uniform(-10, 10, dimensions),
    np.random.uniform(-10, 10, dimensions),
    np.random.uniform(-10, 10, dimensions),
]

# Define MCMC methods to test
mcmc_methods = [
    pints.NoUTurnMCMC,
    pints.RelativisticMCMC,
    pints.MonomialGammaHamiltonianMCMC,
    pints.MALAMCMC,
    pints.HamiltonianMCMC,
]

# Function to run MCMC and plot results
def run_mcmc(method, log_posterior, xs, iterations=4000, warmup=1000, plot=True):
    # Create MCMC routine
    mcmc = pints.MCMCController(log_posterior, len(xs), xs, method=method)
    mcmc.set_max_iterations(iterations)
    mcmc.set_log_to_screen(True)
    mcmc.set_log_interval(100)

    # Run!
    print(f'Running {method.__name__}...')
    chains = mcmc.run()
    print('Done!')

    # Discard warm-up
    chains = chains[:, warmup:]

    if plot:
        # Plot traces and histograms
        pints.plot.trace(chains)
        plt.show()

        # Check convergence and other properties of chains
        results = pints.MCMCSummary(chains=chains[:, 200:], time=mcmc.time(), parameter_names=[f'param_{i}' for i in range(dimensions)])
        print(results)

    return chains

# Run each MCMC method and store results
results = {}
for method in mcmc_methods:
    chains = run_mcmc(method, log_posterior, xs)
    results[method.__name__] = chains

# Optionally, plot pairwise distributions for the first chain of each method
for method_name, chains in results.items():
    pints.plot.pairwise(chains[0], kde=True)
    plt.suptitle(f'Pairwise distribution for {method_name}')
    plt.show()


# Plot the multi-modal distribution
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

for i in range(len(means)):
    diff = np.array([X, Y]).T - means[i]
    Z += weights[i] * np.exp(-0.5 * np.sum(diff @ np.linalg.inv(covariances[i]) * diff, axis=2))

plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar()
plt.title('Multi-Modal Normal Distribution')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
'''
This test performs MCMC using the Metropolis-Hastings algorithm to compute
the expectation of a given function f with respect to a given multivariate
Gaussian pdf.

How to use it: you will probably want to start by playing with the
multivariate_main function, which is where you can define your f and the
parameters of your Gaussian, set the "burn-in" period M, sample run length N,
and number of runs num_runs that you want to use when computing the final
estimate of E[f(X)].

NOTE: unlike in the single_test case, you will NOT need to tune the variance
of the proposal distribution yourself -- this implementation computes it
for you using a simple stochastic approximation scheme to obtain a roughly
23% acceptance rate. This ensure adequate but efficient exploration of the
state space of the Markov chain.
'''

import numpy as np
import scipy.stats as st

import mcmc


def multivariate_test(dim, f, p_mean, p_cov, M, N, init_state, num_runs, \
                      proposal_cov=1.0, calibrate_cov=True):

    p_dist = mcmc.gaussian_pdf(p_mean, p_cov)
    p = p_dist.evaluate
    
    q = mcmc.proposal_dist(
            mcmc.conditional_gaussian(proposal_cov*np.eye(dim)).sample)
    integrator = mcmc.mcmc_integrator(p, q, M, N, init_state)
    integrator.burn()
    
    print("Calibrating variance of the proposal distribution...")
    if calibrate_cov:
        discount = 0.95
        i = 0
        while not (0.225 < integrator.acceptance_rate < 0.235) and i < 100:
            print(proposal_cov)
            proposal_cov -= discount**i*(0.23 - integrator.acceptance_rate)
            proposal_cov = np.clip(proposal_cov, 0.01, 20)
            q = mcmc.proposal_dist(
                    mcmc.conditional_gaussian(proposal_cov*np.eye(dim)).sample)
            integrator = mcmc.mcmc_integrator(p, q, M, N, init_state)
            integrator.burn()
            i += 1

    estimates = []
    print("Variance: %s" % proposal_cov)
    print("Acceptance rate: %s" % integrator.acceptance_rate)
    print("Performing MCMC...")
    for i in range(num_runs):
        new_estimate = integrator.do_mcmc(f)
        estimates.append(new_estimate)
    return estimates


def multivariate_main():
    """Compute the expectation of a user-defined function
    with respect to a multivariate Gaussian random variable with user-defined
    mean and covariance matrix using Markov chain Monte Carlo.
    
    NOTES: As dim gets bigger, M and N must become much larger. The
    covariance matrix for the proposal distribution below is of the simple
    form lambda*np.eye(dim). The program automatically adjusts lambda to
    a suitable value. If you wish to specify your own lambda, set
    proposal_cov=lambda and calibrate_cov=False in the multivariate_test
    function."""
    
    print("Starting the test...")
    
    # Define the function over which the expectation will be taken
    def f(x):
        return np.max(x)
    
    dim = 3                    # dimension of the underlying random variables
    M = 1000                    # burn-in length
    N = 1000                    # length of each sample average
    init_state = np.zeros(dim)  # initial state of the Markov chain
    num_runs = 100              # number of sample averages to compute
    
    # Generate parameters of the multivariate Gaussian over which we'll
    # take the expectation.
    p_mean = np.random.uniform(-1, 1, dim)
    A = np.random.uniform(0, 2, (dim, dim))     # needs to be positive definite
    p_cov = np.dot(A, np.transpose(A))
    
    estimates = multivariate_test(
            dim, f, p_mean, p_cov, M, N, init_state, num_runs,
            proposal_cov=0.5)
    confidence_interval = st.norm.interval(
            0.95, loc=np.mean(estimates), scale=st.sem(estimates))
    
    print("DONE!")
    print("Estimate of E[f(X)]: %s" % np.mean(estimates))
    print("95%% confidence interval for this estimate: %s"
          % np.array(confidence_interval))
    
multivariate_main()
    
    
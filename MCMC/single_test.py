# This test performs MCMC using the Metropolis-Hastings algorithm
# to compute the expectation of a given function f with respect to a given
# pdf p.
#
# How to use it: you will probably want to start by playing with the
# single_main function, which is where you can define your f and p, set
# the "burn-in" period M, sample run length N, and number of runs num_runs
# that you want to use when computing the final estimate of E[f(X)].
#
# NOTE: you will be prompted to choose a suitable value for q_variance, the
# fixed variance of the proposal distribution. The general rule of thumb
# is that you will want a variance that gives you a roughly 23% acceptance
# rate. This ensure adequate but efficient exploration of the state space
# of the Markov chain.

import scipy.stats as st
import numpy as np

import mcmc


def single_test(p, f, q_variance, M, N, init_state, num_runs):
    q_dist = mcmc.conditional_normal(q_variance)
    q = mcmc.proposal_dist(q_dist.sample)
    
    integrator = mcmc.mcmc_integrator(p, q, M, N, init_state)
    integrator.burn()
    
    estimates = []
    for i in range(num_runs):
        estimates.append(integrator.do_mcmc(f))
    
    return estimates


def check_acceptance_rate(p, q_variance):
    q_dist = mcmc.conditional_normal(q_variance)
    q = mcmc.proposal_dist(q_dist.sample)
    
    M = 1000
    N = 5000
    init_state = np.random.uniform(-1,1)
    
    integrator = mcmc.mcmc_integrator(p, q, M, N, init_state)
    integrator.burn()
    # integrator.do_mcmc(f)
    
    return integrator.acceptance_rate
    

def single_main(M, N, init_state, num_runs):
    def p(x):
        return 1/(np.pi*(1+x**2))
    
    def f(x):
        if abs(x) < np.pi:
            return np.cos(x)
        else:
            return 0
        
    q_variance = 1
    good_to_go = False
    
    while not good_to_go:
        acceptance_rate = check_acceptance_rate(p, q_variance)
        print("Current acceptance rate: %s" % acceptance_rate)
        response = input("Enter 'Y' to proceed OR input a new variance: ")
        if response in ['Y', 'y']:
            good_to_go = True
        else:
            try:
                q_variance = float(response)
            except ValueError as error:
                print(error)
                print("q_variance must be a float")
    
    print("Performing MCMC...")
    estimates = single_test(p, f, q_variance, M, N, init_state, num_runs)
    confidence_interval = st.norm.interval(0.95, loc=np.mean(estimates),
                                           scale=st.sem(estimates))
    
    print("DONE!")
    print("Estimate of E[f(X)]: %s" % np.mean(estimates))
    print("95%% confidence interval for this estimate: %s" %
          np.array(confidence_interval))
    
M = 5000
N = 1000
init_state = np.random.uniform(-1,1)
num_runs = 500

single_main(M, N, init_state, num_runs)
    
    
    
    
    
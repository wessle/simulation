'''
Objects for Metropolis-Hastings Markov chain Monte Carlo algorithm.

This module provides some of the basic building blocks needed to use the
Metropolis-Hastings algorithm to compute the expectation of an
arbitrary function with respect to a potentially ugly pdf.  See Ch. 16 in
Spall's Intro to Stochastic Search and Optimization for details).
'''


import numpy as np
from scipy.stats import multivariate_normal


class conditional_normal:
    '''
    Enable sampling from a normal distribution conditional on the mean,
    but with fixed variance.
    '''
    
    def __init__(self, var):
        self.var = var
    
    def sample(self, mean):
        return np.random.normal(loc=mean, scale=self.var)
    
    
class conditional_gaussian:
    '''
    Enable sampling from a multivariate Gaussian distribution conditional
    on the mean, with fixed variance.
    '''
    
    def __init__(self, cov):
        self.cov = cov
        
    def sample(self, mean):
        return np.random.multivariate_normal(mean, cov=self.cov)
    
    
class gaussian_pdf:
    '''
    Create wrapper for multivariate Gaussian pdf with fixed mean and
    covariance.
    '''
    
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
        
    def evaluate(self, x):
        return multivariate_normal.pdf(x, self.mean, self.cov)


class proposal_dist:
    '''
    Conditional distribution of w given x.
    
    The variate_generator is a function taking input x and generating a
    random variate conditioned on x.  The pdf is a function taking inputs w, x
    and outputting the conditional pdf evaluated at w of variate_generator
    conditioned on x.  "symmetric describes" whether the function is
    symmetric or not.
    '''
    
    def __init__(self, variate_generator, pdf=1, symmetric=True):
        self.variate_generator = variate_generator
        self.pdf = pdf
        self.symmetric = symmetric
        
    def sample(self, x):
        return self.variate_generator(x)
    
    def eval_pdf(self, w, x):
        '''
        Automatically returns 1 if the pdf is symmetric.
        
        If you wish to use an asymmetric pdf, you need to pass the pdf in
        separately on initialization.
        '''
        
        if self.symmetric:
            return 1
        else:
            return self.pdf(w, x)


class mcmc_integrator:
    '''
    MCMC integrator.
    
    Use Markov chain Monte Carlo to compute an expectation with respect
    to a potentially nasty pdf p() of a function f() using a proposal 
    distribution q ( | ) with burn-in period M and required number of
    samples N.
    
    See Ch. 16 in Spall's Intro to Stochastic Search
    and Optimization for details.
    '''
    
    def __init__(self, p, q, M, N, init_state):
        self.p = p
        self.q = q
        self.M = M
        self.N = N
        self.init_state = init_state
        self.x = init_state
        
        self.expectation = 0
        self.num_transitions = 0
        self.acceptance_rate = 0
        
    def rho(self, w, x):
        '''
        Compute the probability of accepting w as the next iterate in
        the Markov chain given that the current state is x.
        '''
        
        ratio = self.p(w)/self.p(x) \
            * (self.q.eval_pdf(x, w)/(self.q.eval_pdf(w, x)))
        return min(ratio, 1)
    
    def transition(self):
        '''Carry out transition and update acceptance rate.'''
        
        self.num_transitions += 1
        w = self.q.sample(self.x)
        u = np.random.uniform()
        accepted = 0
        if u < self.rho(w, self.x):
            self.x = w
            accepted = 1
        self.acceptance_rate = ((self.num_transitions-1)*self.acceptance_rate
                                + accepted) / self.num_transitions
        
    def burn(self):
        '''Perform the burn-in to get to steady-state of the Markov chain.'''
        
        for i in range(self.M):
            self.transition()
            
    def do_mcmc(self, f):
        '''Compute the desired expectation using Markov chain Monte Carlo.'''
        
        for i in range(1, self.N+1):
            self.transition()
            self.expectation = ( (i-1) * self.expectation + f(self.x) ) / i
        return self.expectation
    
    def reinit(self):
        '''Reinitialize to the initial state and reset counters, etc. to 0.'''
        
        self.x = self.init_state
        self.expectation = 0
        self.num_transitions = 0
        self.acceptance_rate = 0
        
        
            
            
            
            

import mcmc
import scipy.stats as st
import numpy as np

def single_test(p, f, q_variance, M, N, init_state, num_runs):
    q_dist = mcmc.conditional_normal(q_variance)
    q = mcmc.proposal_dist(q_dist.sample)
    
    integrator = mcmc.mcmc_integrator(p, f, q, M, N, init_state)
    integrator.burn()
    
    estimates = []
    for i in range(num_runs):
        estimates.append(integrator.do_mcmc())
    
    return estimates

def check_acceptance_rate(p, f, q_variance):
    q_dist = mcmc.conditional_normal(q_variance)
    q = mcmc.proposal_dist(q_dist.sample)
    
    M = 1000
    N = 5000
    init_state = np.random.uniform(-1,1)
    
    integrator = mcmc.mcmc_integrator(p, f, q, M, N, init_state)
    integrator.burn()
    integrator.do_mcmc()
    
    return integrator.acceptance_rate
    
def single_main():
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
        acceptance_rate = check_acceptance_rate(p, f, q_variance)
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
    
    M = 5000
    N = 1000
    init_state = np.random.uniform(-1,1)
    num_runs = 500
    
    print("Performing MCMC...")
    estimates = single_test(p, f, q_variance, M, N, init_state, num_runs)
    confidence_interval = st.norm.interval(0.95, loc=np.mean(estimates), \
                                       scale=st.sem(estimates))
    
    print("DONE!")
    print("Estimate of E[f(X)]: %s" % np.mean(estimates))
    print("95%% confidence interval for this estimate: %s" % np.array(confidence_interval))
    
single_main()
    
    
    
    
    
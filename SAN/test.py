import numpy as np
import san

dim = 20
var = 3.0
T = 1000
gamma = 0.98
epoch_length = 100
eps = 0.0001

runs = 5

def projection(point, eps):
    norm = np.linalg.norm(point)
    if norm > eps:
        return (1/norm)*point
    else:
        return point

def runner(dim, var, T, gamma, epoch_length, eps):

    walk = san.random_walk(var*np.eye(dim), dim)
    init_point = np.random.uniform(-20, 20, dim)
    S = san.feasible_region(init_point, walk.step)
    
    def f(x):
        return np.linalg.norm(x-np.ones(dim))**2 * np.linalg.norm(x - 2*np.ones(dim))**2 \
                * np.linalg.norm(x-5*np.ones(dim))**4
                
    def g(x):
        return np.linalg.norm(x-np.ones(dim))**2 * (np.cos(np.linalg.norm(x)) + 2)
    
    annealer = san.annealer(g, S, T, gamma, epoch_length, eps)
    
    print("Initial point: %s" % init_point)
    result = annealer.optimize()
    print("Approximate optimum point: %s" % result)
    print("Approximate optimum value: %s" % annealer.L(result))
    
for i in range(runs):
    runner(dim, var, T, gamma, epoch_length, eps)
    print("\n")
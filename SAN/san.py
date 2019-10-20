'''
Object definitions for simulated annealing with user-specified temperature
epochs and geometric annealing schedule (i.e. T_k+1 = gamma T_k for some
0 < gamma < 1)

NOTE: this performs minimization of the objective function over the
feasible region. The random_walk object is what I'm currently using for
the perturbation, but a noisy gradient would likely be far more efficient.
Both of these are only for continuous settings, of course, so a totally
different perturbation method will need to be defined to handle
combinatorial problems.
'''

import numpy as np


class random_walk:
    def __init__(self, cov, size):
        self.cov = cov
        self.size = size
        
    def step(self, curr_point):
        step = np.random.multivariate_normal(np.zeros(self.size), self.cov)
        return curr_point + step


class feasible_region:
    def __init__(self, init_point, new_point_generator, projection=None):
        self.curr_point = init_point
        self.new_point_generator = new_point_generator
        
    def get_candidate_point(self):
        return self.new_point_generator(self.curr_point)
    
    def get_curr_point(self):
        return self.curr_point
    
    def set_curr_point(self, point):
        self.curr_point = point


class annealer:
    def __init__(self, objective_function, feasible_region, init_temp,
                 gamma, epoch_length, eps):
        self.L = objective_function
        self.feas_reg = feasible_region
        self.T = init_temp
        self.gamma = gamma
        self.epoch_length = epoch_length
        self.eps = eps
        
    def boltz_gibbs(self, delta):
        return np.exp(-delta/self.T)
    
    def run_epoch(self):
        curr_val = self.L(self.feas_reg.get_curr_point())
        for i in range(self.epoch_length):
            candidate_point = self.feas_reg.get_candidate_point()
            delta = self.L(candidate_point) - curr_val
            if delta < 0:
                self.feas_reg.set_curr_point(candidate_point)
            elif np.random.uniform() < self.boltz_gibbs(delta):
                self.feas_reg.set_curr_point(candidate_point)
            else:
                continue
    
    def annealing_update(self):
        self.T *= self.gamma
        
    def optimize(self):
        while self.T > self.eps:
            self.run_epoch()
            self.annealing_update()
            
        return self.feas_reg.get_curr_point()
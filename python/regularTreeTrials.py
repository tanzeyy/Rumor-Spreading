import scipy.io as io
import numpy as np
from scipy import stats
import utilities
import runExperiments

'''Run the irregular tree algorithm'''
if __name__ == "__main__":

    # max_times = [40, 15, 9, 8]
    # degrees = [i for i in range(2,6)]
    max_times = [50, 15]
    degrees = [2, 3]
    trials = 10000
    
    for (degree, max_time) in zip(degrees, max_times):
        max_infection = degree - 1
        
        # Irregular infinite graph
        # xk = np.arange(3,5)
        # pk = (0.5,0.5)
        xk = np.array([degree])
        pk = (1.0)
        
        # want to find pd using the old spreading algorithm
        
        num_infected_all = []
        pd_ml_all = []
        
        degrees_rv = stats.rv_discrete(name='rv_discrete', values=(xk, pk))
        num_infected, pd_rc, pd_jc = runExperiments.run_randtree(trials, max_time, max_infection, degrees_rv, method = 3)
        print('Rumor centrality', pd_rc)
        print('Jordan centrality', pd_jc)
        print('Num infected', num_infected)
        
        filename = 'results/regular_tree_results_d_' + str(degree) + '.mat'
        io.savemat(filename, {'pd_rc':np.array(pd_rc), 'pd_jc':np.array(pd_jc), 'num_infected':np.array(num_infected)})
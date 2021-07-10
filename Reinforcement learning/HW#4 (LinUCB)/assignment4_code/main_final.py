# %%

import numpy as np
from algorithms import LinUCB, RegretBalancingElim
from utils import FiniteContLinearRep, ContBanditProblem, make_random_rep, make_newlinrep_reshaped
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed

L = []
def execute_regbalelim(T, true_rep, reps, reg_val, noise_std, delta, num):
    algo = RegretBalancingElim(reps, reg_val, noise_std, delta)
    #algo = RegretBalancingElim2(reps, reg_val, noise_std, delta)
    problem = ContBanditProblem(true_rep, algo, noise_std)
    start_time = time.time()
    problem.run(T)
    #print(f"--- {num} finished in {np.round(time.time() - start_time,2)} seconds ---")
    reg = problem.exp_instant_regret.cumsum()
    print("RegBalELim active repressentations are num:",algo.active_reps)
    L.append(algo.active_reps)
    return reg

def execute_linucb(T, true_rep, rep, reg_val, noise_std, delta, num):
    algo = LinUCB(rep, reg_val, noise_std, delta)
    problem = ContBanditProblem(true_rep, algo, noise_std)
    start_time = time.time()
    problem.run(T)
    print(f"--- {num} finished in {np.round(time.time() - start_time,2)} seconds ---")
    reg = problem.exp_instant_regret.cumsum()
    return reg

PARALLEL = True
NUM_CORES = 5
NRUNS = 5
nc, na, nd = 200, 20, 20
noise_std = 0.3
reg_val = 1
delta = 0.01
#T = 50000
T = 50000


b = np.load('final_representation2.npz')
reps = []
for i in range(b['true_rep']+1):
    feat = b[f'reps_{i}']
    param = b[f'reps_param_{i}']
    reps.append(FiniteContLinearRep(feat, param))
linrep = reps[b['true_rep']]

# print("Running algorithm RegretBalancingElim")
# results = []
# if PARALLEL:
#     results = Parallel(n_jobs=NUM_CORES,backend='multiprocessing')(
#         delayed(execute_regbalelim)(T, linrep, reps, reg_val, noise_std, delta, i) for i in range(NRUNS)
#     )
# else:
#     for n in range(NRUNS):
#         results.append(
#             execute_regbalelim(T, linrep, reps, reg_val, noise_std, delta, n)
#         )
# regrets = []
# for el in results:
#     regrets.append(el.tolist())
# regrets = np.array(regrets)
# mean_regret = regrets.mean(axis=0)
# std_regret = regrets.std(axis=0) / np.sqrt(regrets.shape[0])
# plt.plot(mean_regret, label="RegBalElim")
# plt.fill_between(np.arange(T), mean_regret - 2*std_regret, mean_regret + 2*std_regret, alpha=0.1)

print("Running algorithm LinUCB")
#reps_good = [reps[b['true_rep']]]
reps_good= [reps[2],reps[7]]
rep_num = [2,7]
#for nf, f in enumerate(reps):
for nf, f in zip(rep_num,reps_good):

    results = []
    if PARALLEL:
        results = Parallel(n_jobs=NUM_CORES)(
            delayed(execute_linucb)(T, linrep, f, reg_val, noise_std, delta, i) for i in range(NRUNS)
        )
    else:
        for n in range(NRUNS):
            results.append(
                execute_linucb(T, linrep, f, reg_val, noise_std, delta, n)
            )

    regrets = []
    for el in results:
        regrets.append(el.tolist())
    regrets = np.array(regrets)
    mean_regret = regrets.mean(axis=0)
    std_regret = regrets.std(axis=0) / np.sqrt(regrets.shape[0])
    plt.plot(mean_regret, label=f"LinUCB - f{nf}")
    plt.fill_between(np.arange(T), mean_regret - 2*std_regret, mean_regret + 2*std_regret, alpha=0.1)

plt.legend()
plt.show()

#%%
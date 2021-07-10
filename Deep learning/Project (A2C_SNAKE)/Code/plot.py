import matplotlib.pyplot as plt
import os


path = '/home/mahdo/Desktop/deep_final/RL-Snake/Experiments/Experiments_bs'
files = []
for i in os.listdir(path):
    
    if os.path.isfile(os.path.join(path,i)) and 'rewards' in i:
        files.append(i)

for i,fil in enumerate(files):

     exp = np.loadtxt(fil)
        
            plt.figure()
            plt.title('Mean reward over learning')
            plt.plot(exp[:,0], exp[:,2])
            plt.fill_between(exp[:,0], exp[:,2]+exp[:,3] , exp[:,2]-exp[:,3] , alpha = 0.2)
            plt.xlabel('training steps')
            
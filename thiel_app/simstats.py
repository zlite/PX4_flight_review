# this is the module that does the sim-to-real comparison stats. Feel free to write your own!

import numpy as np
import math

def sim2real_stats(data):
    simdata = data['sim']
    realdata = data['real']
    simdata.dropna(inplace=True)  # delete empty rows
    realdata.dropna(inplace=True)  
    real = np.array(realdata)
    sim = np.array(simdata)
    stats = 0
    if (len(real) != 0) and (len(sim) !=0):  # avoid divide by zero errors
        sum1 = 0
        sum2 = 0
        sum3 = 0
        for n in range(len(real)):
            sum1 = sum1 + (real[int(n)]-sim[int(n)])**2
            sum2 = sum2 + real[int(n)]**2
            sum3 = sum3 + sim[int(n)]**2
        sum1 = 1/len(real) * sum1
        sum2 = 1/len(real) * sum2
        sum3 = 1/len(real) * sum3
        sum1 = math.sqrt(sum1)
        sum2 = math.sqrt(sum2)
        sum3 = math.sqrt(sum3)
        stats = sum1/(sum2 + sum3)
        stats = round(stats,3)
    return stats
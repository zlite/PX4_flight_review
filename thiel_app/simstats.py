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
        length = min(len(real), len(sim))  # only compare over the overlapping range of the two datasets, which is the smallest of the two
        for n in range(length):
            sum1 = sum1 + (real[int(n)]-sim[int(n)])**2
            sum2 = sum2 + real[int(n)]**2
            sum3 = sum3 + sim[int(n)]**2
        sum1 = 1/length * sum1
        sum2 = 1/length * sum2
        sum3 = 1/length * sum3
        sum1 = math.sqrt(sum1)
        sum2 = math.sqrt(sum2)
        sum3 = math.sqrt(sum3)
        stats = sum1/(sum2 + sum3)
        stats = round(stats,3)
    return stats

def sim2real_stats2(data):  # this is the Song variation of Thiel: https://drive.google.com/file/d/1XY8aZz89emFt-LAuUZ2pjC1GHwRARr9f/view
    numerator = 0
    denominator = 0
    simdata = data['sim']
    realdata = data['real']
    simdata.dropna(inplace=True)  # delete empty rows
    realdata.dropna(inplace=True)  
    real = np.array(realdata)
    sim = np.array(simdata)
    length = min(len(real), len(sim))  # only compare over the overlapping range of the two datasets, which is the smallest of the two
    for i in range(length-1):
        numerator = numerator + ((real[i+1]-real[i])-(sim[i+1]-sim[i]))**2
        denominator = denominator + (real[i+1] - real[i])**2
    
    numerator = math.sqrt(numerator)
    denominator = math.sqrt(denominator)
    total = round(numerator/denominator,3)
    return total
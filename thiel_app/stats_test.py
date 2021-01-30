import simstats 


# sample data

real = [1,2,3,4,5,6,5,4,3,2,1]
sim = [3,4,5,6,7,6,5,4,3,2,1]

score = simstats.equation_8(real, sim)
print("score = ", score)
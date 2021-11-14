# Travelling Salesman Problem
### Implemented a genetic algorithm that experimentally obtains a near-optimal solution to the classic Travelling Salesman Problem.

In this classic optimization problem, the objective function is to minimize the distance travelling between nodes, with the decision variable being the order of cities. The constraints include: 

 1) each city have to be visited exactly once, and 
 2) the salesman have to return to the first city.

Our case in context is the United States.

![alt text](https://github.com/christopherfkk/Travelling-Salesman-Problem/blob/main/tsp_genetic.png)

We define the chromosomes as lists of integers that represent the cities. Survival, mutation, and crossover, introduce stochasticity to expand the search space so that a global optimum is more likely to be reached. The fitness is graphed at every generation (i.e. iteration) and the final path is obtained after a fixed number of generations (i.e. 100 iterations). 


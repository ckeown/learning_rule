#!/usr/bin/env python2.7
#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import array
import multiprocessing
import random
import sys

if sys.version_info < (2, 7):
    print("mpga_onemax example requires Python >= 2.7.")
    exit(1)

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from NeuralNetwork import *


creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# Tells it to maximize the fitness
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("attr_float", random.random)

# Structure initializers
# Specifies the number of "genes" and that they'll be floats.
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 20)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# This defines your evaluation method and return the fitness(es)
def evalOneMax(individual):

    # create the network and connect the nodes
    # this is current set up as a feed forward network
    net = Network(13,7,4)
    net.connect_all()
    
    # probably an unnecessary step...
    coeffs = [individual[x] for x in range(20)]
    

    # So as to not repeat things, you can read about the format of the dataset on the webpage
    # https://archive.ics.uci.edu/ml/machine-learning-databases/00195/documentation.html
    # For the training dataset, the first 660 blocks are for the number 0, the second 660 are for the number 1, etc.
    
    
    # in this loop, we are setting the weights
    i = 0
    for (timeseries,output) in training_data:
        j = 0
        for timepoint in timeseries:
            net.evaluate(timepoint)
            j += 1
            if j > 20:
                # after 20 time steps, it will start updating the weights using the proposed training rule
                # yes, with the current topology being just feedforward, the first 20 timesteps could just be skipped.
                net.update_output_layer_weights(output, coeffs)
                # we'll want to uncomment this eventually, but I was just playing with updating output layer
                # net.update_hidden_layer_weights(coeffs)

        i += 1
        # tell it to stop after the first letter (another nonsensical step)
        if i >= 660: break


    # Now we're computing the fitness by using the same dataset and just seeing how well
    # the weights updated to it. There's also a separate test dataset that has blocks
    # of size 220 (as opposed to 660). That is probably better used as a validation dataset.
    correct = 0.0
    total = 0.0
    i = 0
    for (timeseries,output) in training_data:
        # runs the data through the next (with FF net only need to run the last time point)
        for timepoint in timeseries:
            net.evaluate(timepoint)
        net_output = net.get_output()
        error = sum([abs(output[x] - net_output[x]) for x in range(len(output))])        
        # allow some margin of error for being correct
        if error < 0.5:
            correct += 1
        total += 1
        i += 1
        if i >= 660: break

    # random thoughts...
    # Is it worthwhile just to have an output for each posibility and max sure the maximum is correct??
    # try to grow the genome over time, not just all at once.
    # bias node??

    # make sure to return that comma (per the documentation)!!
    return correct/total, 


# This section add mutation, crossover, and the evaluation function (the one you have to write).
toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Just put some bounds on your values
def checkBounds(min, max):
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in xrange(len(child)):
                    if child[i] > max:
                        child[i] = max
                    elif child[i] < min:
                        child[i] = min
            return offspring
        return wrapper
    return decorator

toolbox.decorate("mate", checkBounds(-1.0, 1.0))
toolbox.decorate("mutate", checkBounds(-1.0, 1.0))


if __name__ == "__main__":
    random.seed(64)
    
    # Process Pool of 4 workers
    pool = multiprocessing.Pool(processes=4)
    toolbox.register("map", pool.map)
    
    # This line sets the population size.
    pop = toolbox.population(n=10)
    hof = tools.HallOfFame(1)
    # These lines tell it what to print after each iteration.
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    # This line really calls the GA. ngen = number of generations.
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, 
                        stats=stats, halloffame=hof)

    # new to write some code here to print out the individual genome
    # so we can see which coefficients are found.

    pool.close()

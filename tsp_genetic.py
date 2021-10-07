# Import relevant libraries
import matplotlib.pyplot as plt
import random
import numpy as np
import networkx as nx

### Step 1: initializing population

def generate_uniform_map(plot = True):
    
    # Dict: coordinates of the aforementioned US cities [latitude,longitude]
    cities = {
        "1": [47.4502, 122.3088], # seattle
        "2": [45.5898, 122.5951], # portland
        "3": [33.9416, 118.4085], # los angeles
        "4": [36.0840, 115.1537], # las vegas
        "5": [40.7899, 111.9791], # salt lake city
        "6": [33.4484, 112.0740], # phoenix
        "7": [39.8561, 104.6737], # denver
        "8": [35.0433, 106.6129], # albuquerque
        "9": [39.3036, 94.7093], # kansas city
        "10": [29.9902, 95.3368], # houston
        "11": [41.9803, 87.9090], # chicago
        "12": [33.6407, 84.4277], # atlanta
        "13": [27.9772, 82.5311], # florida
        "14": [38.9531, 77.4565], # washington
        "15": [40.6413, 73.7781] # new york
    }
    
    # creates two separate lists storing the N and W coordinates for plotting
    xs = []
    ys = []
    for i in range(1,16):
        xs.append(cities[str(i)][1])
    for i in range(1,16):
        ys.append(cities[str(i)][0])

    if plot:
        plt.scatter(xs, ys, s = 200, c = 'darkmagenta')
        plt.xlabel('X coordinate of the city')
        plt.ylabel('Y coordinate of the city')
        plt.show()

    return cities

# returns one chromosome: a route, defined by a list of cities
def random_route(citylist):
    
    #dict.keys() returns the keys of the dictionary, i.e. the node numbers of cities
    keys = list(citylist.keys())
    np.random.shuffle(keys)
    return keys

# returns our population: a list of routes
def initialize_routes(population_size, citylist):
    population = []
    for i in range(population_size):
        population.append(random_route(citylist))
    return population

### step 2: fitness function calculation - the distance of routes

# returns distance between two coordinates based on Pythagoras' theorem
def get_distance(citylist, indexi, indexj):
    city_a = citylist[indexi]
    city_b = citylist[indexj]
    
    # [latitude (vertical; y),longitude(horizonta; x))] 
    x_dis = abs(city_a[1] - city_b[1]) * 111 # one latitude degree is ~111 km
    y_dis = abs(city_a[0] - city_b[0]) * 111 # one longitude degree is ~111 km
    distance = np.sqrt((x_dis ** 2) + (y_dis ** 2))
    return distance

# returns the total distance of one route
def get_route_length(citylist, route):
    total_dist = 0
    for index in range(len(route)):
        if index < len(route) - 1:
            dist = get_distance(citylist, route[index], route[index+1])
        else:
            dist = get_distance(citylist, route[index], route[0])
        total_dist += dist
    return total_dist

# returns a sorted list of distances (our fitness measure) by ascending order
def rank_routes(citylist, population):
    fitnessresults = []
    
    # appends all the route lengths to fitnessresults
    for index in range(len(population)):
        fitnessresults.append([index, get_route_length(citylist, population[index])])

    # sorts by route length, or position 1 of an item in fitnessresults
    fitnessresults.sort(key=lambda x: x[1])
    return fitnessresults  

### step 3: survivor selection

# returns a list of routes that survived
def get_crossover_pool(fitnessresults, population_size, survival_rate):
    
    cross_pool = []
    
    # determines the number of chromosomes that will survive
    elite_amount = int(population_size * survival_rate)
    
    # places direct survivors into the pool
    for route in fitnessresults[0:elite_amount]:
        cross_pool.append(route[0])

    return cross_pool

### step 4: crossover or breeding using the survived chromosomes

# returns two children from two parents based on Davis' order crossover
def davis_crossover(parent1, parent2):
    
    # sets maximum length of children for the while loops later
    num_cities = len(parent1)
    
    # initializes two empty lists as the two children 
    child_1 = []
    child_2 = []
    
    # randomly generates two break points
    breakpoints = np.sort(np.random.choice(num_cities,2,replace=False))
    
    # child 1 receives the segment in parent 1 that has range (first break point, second break point)
    # child 2 receives the segment in parent 2 that has range (first break point, second break point)

    for i in range(breakpoints[0],breakpoints[1]):
        child_1.append(parent1[i])
        child_2.append(parent2[i])
    sortpoint = num_cities - breakpoints[0]
    j = breakpoints[1]
    
    # child 1 receives the remainder of parent 2 and wraps before and after the range of segment
    while len(child_1) < num_cities:
        while parent2[j%num_cities] in child_1:
            j += 1
        child_1.append(parent2[j%num_cities])
    child_1 = child_1[sortpoint:] + child_1[:sortpoint]
    j = breakpoints[1]
    
     # child 2 receives the remainder of parent 1 and wraps before and after the range of segment
    while len(child_2) < num_cities:
        while parent1[j%num_cities] in child_2:
            j += 1
        child_2.append(parent1[j%num_cities])
    child_2 = child_2[sortpoint:] + child_2[:sortpoint]
    
    return [child_1, child_2]

# returns the children of the next generation consisting of survived parents and breeded children
def population_crossover(population_size, population, cross_pool):
    children = []

    # make the direct survivors an even number so all breeded children can be include
    if len(cross_pool) % 2 != 0:
        cross_pool.pop()

    # put the direct survivors into the next generation
    for routeindex in cross_pool:
        children.append(population[routeindex])

    # while the next generation is under the population size, keep breeding until it reaches maximum
    while len(children) < population_size:
        
        # find two random parent to breed two children
        sampleindex = np.random.choice(cross_pool, 2, replace = False)
        parent1 = population[sampleindex[0]]
        parent2 = population[sampleindex[1]] 
        children += davis_crossover(parent1, parent2)
    return children 

### step 5: mutation

# mutate by randomly swapping two cities
def mutate(route, mutation_rate, seed = 2021):
    for swapped in range(len(route)):
        
        # np.random.random() geneates random number from 0-1 with equal probabilities
        # if mutation rate is 1%, 1% of the time the random number is under 0.01
        # if random number is above 0.01, no swaps happen
        if(np.random.random() < mutation_rate):
            
            # generates a random index of the route list to swap with
            swapwith = int(np.random.random() * len(route))
            route[swapped], route[swapwith] = route[swapwith], route[swapped]
            
    return route

def mutate_population(mutation_rate, children):
    mutated_routes = []

    # iterates through 100% of the population and mutates a small portion of the population
    for route in children:
        mutated_route = mutate(route, mutation_rate)
        mutated_routes.append(mutated_route)

    return mutated_routes

### step 6: visualize the genetic algorithm

def visualize(citylist, population, fitnessresults):
    
    # creates graph with network moduel
    graph = nx.Graph()
    
    # Adding nodes (numbers_ to the graph from the diction "cities"
    graph.add_nodes_from(citylist.keys())
    
    # gets the best route at position[0][0] because it was previously sorted in ascending order of distance or descending order of fitness
    if fitnessresults:
        best_route = fitnessresults[0][0]
        
        # adds edges to the best route for visualization
        for node_index in range(len(population[best_route]) - 1):
            graph.add_edge(str(population[best_route][node_index]), str(population[best_route][node_index+1]))
        graph.add_edge(str(population[best_route][-1]), str(population[best_route][0]))

    # creates the plot
    pos = {}
    plt.figure(figsize=(6.5,6.5))
    for node in citylist.keys():
        pos[node] = tuple(citylist[node])
    nx.drawing.nx_pylab.draw_networkx(graph,align="horizontal", pos=pos,font_color='white',\
        font_weight='bold',font_size=11,node_color='darkmagenta',node_size=300)
    plt.show()

# The function below chooses what generations to display
# The routes change quickly at first, and then more slowly as the evolution progresses.
# Thus, this function chooses to look at generations that are closer together 
# earlier in the evolution, becoming more spaced out as we go along.

def get_displays(n_displays, n_plots, n_generations):
    displayed_gens = [1+np.floor((n_generations-1)*(i/n_displays)**2) for i in range(0, n_displays+1)]
    displayed_plots = [1+np.floor((n_generations-1)*(i/n_plots)**2) for i in range(0, n_plots+1)]
    return displayed_gens, displayed_plots

### step 7: define parameters

# Define the parameters of our genetic algorithm
cities_count = 20
population_size = 50
n_generations = 100
mutation_rate = 0.02 # low mutation rate because a higher value can lead to random search
survival_rate = 0.6

## #step 8: run the model and call all the defined functions

# Initialize the population of routes based on the map of US
cities = generate_uniform_map() 
population = initialize_routes(population_size, cities)
fitnessresults = rank_routes(cities, population)

# Get 3 generations to display
default_displays = 3
display = get_displays(default_displays,default_displays,n_generations)
fitness_evolution = np.zeros(n_generations)

# Main loop of the algorithm
for gen in range(1, n_generations+1):
    
    # displays the route and its associated fitness value
    if gen in display[0]:
        print(f'Generation: {gen}\nBest distance: {str(round(fitnessresults[0][1],0))} km')
    if gen in display[1]:
        visualize(cities, population, fitnessresults)
    cross_pool = get_crossover_pool(fitnessresults, population_size, survival_rate)
    
    # if direct survivors less than 2, there isn't at least a pair of parents that can breed
    # breaks the algorithm 
    if len(cross_pool)<2:
        print("Not enough parents to produce children!")
        print("For a population of size",population_size,", the survival rate must be at least",2/population_size)
        break
    else:
        children = population_crossover(population_size, population, cross_pool)
        population = mutate_population(mutation_rate, children)
        fitnessresults = rank_routes(cities, population)
        fitness_evolution[gen-1] = fitnessresults[0][1]

# plot to show fitness over the generations
if len(cross_pool)>=2:
    plt.plot(list(range(n_generations)),fitness_evolution,c = 'seagreen',linewidth=2)
    plt.title('Best route length per generation', fontsize = 14)
    plt.xlabel('Generation number', fontsize = 12)
    plt.ylabel('Best route length', fontsize = 12)
    plt.show()
    print("Best path in the final generation:\n",population[0],"\nDistance: ",round(fitnessresults[0][1],0), "km")

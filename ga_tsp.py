
# coding: utf-8

# In[3]:


import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt


# ## Create necessary classes and functions

# In[4]:


xl=pd.ExcelFile('./data/CA_TSP.xlsx')
print(xl.sheet_names)
cities = xl.parse('city_ridership')
cities_dist_matrix = xl.parse('city_dist_matrix')


# Create class to handle "cities"

# In[5]:

class City:
    def __init__(self,name,population,has_petrol):
        self.name=name
        self.population=population
        self.has_petrol=has_petrol
    
    def distance(self, city):
        distance=cities_dist_matrix.loc[self.name,city.name]
        return distance
    
    def __repr__(self):
        return self.name+"["+str(self.population)+"]"


# Create a fitness function

# In[6]:


class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness= 0.0
        self.total_population= 0
        self.maxDistFuel = 0
        self.limitFuelDistance=200
        self.feasibility=0
    
    def routeDistance(self):
        pathDistance = 0
        for i in range(0, len(self.route)):
            fromCity = self.route[i]
            toCity = None
            if i + 1 < len(self.route):
                toCity = self.route[i + 1]
            else:
                break
            pathDistance += fromCity.distance(toCity)
        self.distance = pathDistance
        return self.distance

    def maxDistBWFuel(self):
        maxDistance = 0
        distance = 0
        for j in range(0,len(self.route)-1):
            fromCity=self.route[j]
            toCity=self.route[j+1]
            if(toCity.has_petrol==1):
                distance += fromCity.distance(toCity)
                maxDistance=distance if distance > maxDistance else maxDistance
                distance=0
            elif(j+1==len(self.route)-1):
                distance += fromCity.distance(toCity)
                maxDistance = distance if distance > maxDistance else maxDistance
                distance = 0
            else:
                distance += fromCity.distance(toCity)
                maxDistance = distance if distance > maxDistance else maxDistance
        self.maxDistFuel = maxDistance
        return self.maxDistFuel
    
    def routePopulation(self):
        path_population = 0
        for i in range(0, len(self.route)):
            City = self.route[i]
            path_population += City.population
        self.total_population = path_population
        return self.total_population
    
    def routeFitness(self):
        if self.maxDistBWFuel() > self.limitFuelDistance:
            self.feasibility=0
        else:
            self.feasibility=1

        self.fitness = self.routePopulation() - float(self.routeDistance()) if self.feasibility == 1 else -1000000.00
        return self.fitness


# ## Create our initial population

# Route generator

# In[7]:


def createRoute(cityList):
    size=random.randint(7, len(cityList))
    route = random.sample(cityList, 10)
    return route


# Create first "population" (list of routes)

# In[8]:


def initialPopulation(popSize, cityList):
    population = []
    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population


# ## Create the genetic algorithm

# Rank individuals

# In[9]:


def rankRoutes(population):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)


# Create a selection function that will be used to make the list of parent routes

# In[10]:


def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults


# Create mating pool

# In[11]:


def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool


# Create a crossover function for two parents to create one child

# In[12]:


def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    '''
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])
        
    childP2 = [item for item in parent2 if item not in childP1]

    
    '''
    for i in range(0,len(parent1)):
        if(i<geneA):
            childP1.append(parent1[i])
    i=geneA
    for item in parent2:
        if item not in childP1 and i < len(parent1):
            childP2.append(item)
            i+=1
    ''''''

    child = childP1 + childP2
    return child


# Create function to run crossover over full mating pool

# In[13]:


def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0,eliteSize):
        children.append(matingpool[i])
    
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children


# Create function to mutate a single route

# In[14]:


def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))
            
            city1 = individual[swapped]
            city2 = individual[swapWith]
            
            individual[swapped] = city2
            individual[swapWith] = city1
    return individual


# Create function to run mutation over entire population

# In[15]:


def mutatePopulation(population, mutationRate):
    mutatedPop = []
    
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop


# Put all steps together to create the next generation

# In[16]:


def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration


# Final step: create the genetic algorithm

# In[17]:


def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(Fitness(pop[rankRoutes(pop)[0][0]]).routeDistance()))
    print("Initial population: " + str(Fitness(pop[rankRoutes(pop)[0][0]]).routePopulation()))
    print("Feasibility: " + str(Fitness(pop[rankRoutes(pop)[0][0]]).feasibility))
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)

    total_distance=0
    total_population=0
    for i in range (0,10):
        bestRouteIndex = rankRoutes(pop)[i][0]
        bestRoute = pop[bestRouteIndex]
        fitness = Fitness(bestRoute)
        fitness.routeFitness()
        print((i+1),"route:", bestRoute)
        total_distance += fitness.distance
        print("distance: " + str(fitness.distance))
        total_population += fitness.total_population
        print("Population: " + str(fitness.total_population))
        print("feasibility: " + str(fitness.feasibility))

    print("Total distance= "+ str(total_distance))
    print("Total population= " + str(total_population))
    return bestRoute


# ## Running the genetic algorithm

# Create list of cities

# In[ ]:


cityList = []
len(cities.City)
for i in range(0,len(cities.City)):
    cityList.append(City(name=cities.City.iloc[i],population=cities.Population.iloc[i],has_petrol=cities.Has_petrol.iloc[i]))


# Run the genetic algorithm

# In[ ]:


geneticAlgorithm(population=cityList, popSize=600, eliteSize=80, mutationRate=0.01, generations=1000)


# ## Plot the progress

# Note, this will win run a separate GA

# In[ ]:


def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    progress = []
    progress.append(Fitness(pop[rankRoutes(pop)[0][0]]).routeDistance())
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progress.append(Fitness(pop[rankRoutes(pop)[0][0]]).routeDistance())
    
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()


# Run the function with our assumptions to see how distance has improved in each generation

# In[ ]:


#geneticAlgorithmPlot(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)


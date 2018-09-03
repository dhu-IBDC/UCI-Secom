#This Python file uses the following encoding: utf-8

from deap import base, creator
import random
import numpy as np
from deap import tools
import fitness_function as ff


class FeatureSelectionGA:
    def __init__(self, model, x, y, x_test,y_test,x_development,y_development, verbose=1):
        self.model = model
        self.n_features = x.shape[1]
        self.toolbox = None
        self.creator = self._create()
        self.x = x
        self.y = y
        self.x_test=x_test
        self.y_test=y_test
        self.x_development=x_development
        self.y_development=y_development
        self.verbose = verbose
        if self.verbose==1:
            print("Model {} will select best features among {} features .".format(model,x.shape[1]))
            print("Shape od train_x: {} and target: {}".format(x.shape,y.shape))
        self.final_fitness = []
        self.fitness_in_generation = {}
        self.best_ind = None

    def evaluate(self, individual):
        fit_obj = ff.FitenessFunction()
        np_ind = np.asarray(individual)
        if np.sum(np_ind) == 0:
            fitness = 0.0
        else:
            feature_idx = np.where(np_ind == 1)[0]
            fitness = fit_obj.calculate_fitness\
                (self.model,self.x[:,feature_idx], self.y, self.x_test[:,feature_idx],self.y_test,self.x_development[:,feature_idx],self.y_development)
        if self.verbose == 1:
            pass
        return fitness,
    
    
    def _create(self):
        creator.create("FeatureSelect", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FeatureSelect)
        return creator
    

        
    def register_toolbox(self, toolbox):
        toolbox.register("evaluate", self.evaluate)
        self.toolbox = toolbox
     
    
    def _init_toolbox(self):
        toolbox = base.Toolbox()
        toolbox.register("attr_bool", random.randint, 0, 1)
        # Structure initializers
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, self.n_features)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        return toolbox
        
        
    def _default_toolbox(self):
        toolbox = self._init_toolbox()
        toolbox.register("mate", tools.cxOnePoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
        toolbox.register("select", tools.selTournament,tournsize=3)
        toolbox.register("evaluate", self.evaluate)
        return toolbox
    
    def get_final_scores(self ,pop ,fits):
        self.final_fitness = list(zip(pop,fits))
    
        
    def generate(self,n_pop,cxpb = 0.9,mutxpb = 0.4,ngen=20,set_toolbox = False):
        if self.verbose == 1:
            print("Population: {}, crossover_probablity: {}, mutation_probablity: {}, total generations: {}".format(n_pop,cxpb,mutxpb,ngen))
        
        if not set_toolbox:
            self.toolbox = self._default_toolbox()
        else:
            raise Exception("Please create a toolbox.Use create_toolbox to create and register_toolbox to register. Else set set_toolbox = False to use defualt toolbox")
        pop = self.toolbox.population(n_pop)
        CXPB, MUTPB, NGEN = cxpb,mutxpb,ngen

        # Evaluate the entire population
        print("EVOLVING.......")
        fitnesses = list(map(self.toolbox.evaluate, pop))
        
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        for g in range(NGEN):
            print("-- GENERATION {} --".format(g+1))
            offspring = self.toolbox.select(pop, len(pop))
            self.fitness_in_generation[str(g+1)] = max([ind.fitness.values[0] for ind in pop])
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            for mutant in offspring:
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            # Evaluate the individuals with an invalid fitness
            weak_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, weak_ind))
            for ind, fit in zip(weak_ind, fitnesses):
                ind.fitness.values = fit
            print("Evaluated %i individuals" % len(weak_ind))
            pop[:] = offspring
        fits = [ind.fitness.values[0] for ind in pop]
        print("-- Only the fittest survives --")
        self.best_ind = tools.selBest(pop, 1)[0]
        print("Best individual is %s, " % self.best_ind)
        print('The best recall is:%s'%self.best_ind.fitness.values)
        self.get_final_scores(pop,fits)
        f_obj=ff.keshihua()
        np_ind = np.asarray(self.best_ind)
        feature_idx = np.where(np_ind == 1)[0]
        fitness = f_obj.huatu \
            (self.model, self.x[:, feature_idx], self.y, self.x_test[:, feature_idx], self.y_test,
             self.x_development[:, feature_idx], self.y_development)
        
        return pop


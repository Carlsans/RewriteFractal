import os
import sys

import optuna

from src.NormVar import NormVar
from src.NormalizedFunctions import strange_distance

print(sys.path)
sys.path.append('/home/carl/PycharmProjects/RewriteFractal')
from src.functions.Trigonometric import Trigonometric
from src.StudyManager import StudyManager
import json
import time
from collections import OrderedDict
import jsonpickle
import numpy as np
import torch
from deap import base, creator, tools, algorithms
import random
import string
import copy
random.seed(time.time() + os.getpid())

from src.Configuration import Configuration
from src.Fractal import Fractal
from src.PoolManager import PoolManager
from src.ClaudesMethod import ImageComplexityMetrics
from src.GAHyperParameters import GAHyperParameters
# noinspection PyUnresolvedReferences
#from RewriterEngineC import RewriterEngine
from src.RewriterEngine import RewriterEngine
from src.RewriterEvaluator import RewriterEvaluator
from src.RewriterRandomGenerator import RewriterRandomGenerator
from src.Sieve import Sieve
from RewriterEngineCython import RewriterEngineCython


class GeneticAlgorithm:

    def __init__(self):
        self.conf = Configuration()
        self.hp = GAHyperParameters()
        self.sieve = Sieve()
        self.fractal = Fractal(self.sieve,center=(0,0),range=2,useexitbailoutvalues=True)
        self.trigo = Trigonometric()
        self.ic = ImageComplexityMetrics()
        print("new ga")
        self.poolman = PoolManager(self.conf)
        # Create fitness and individual classes
        self.creator = creator
        self.creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        self.creator.create("Individual", list, fitness=self.creator.FitnessMin)
        #self.vocabulary = "ACGTZa"
        self.operators = "+-*/"

        self.rwevaluator = RewriterEvaluator(self.conf)
        self.rwc = RewriterEngineCython()
        self.absolute_best_fitness = 100000000000000000
        self.generations_without_improvement =  0
        self.minentropy = 1000000000000
        self.maxentropy =0
    def generate_individual(self):

        if False:# Here we force branching ;)random.random() > self.hp.fetchbestpool:
            individual = self.poolman.getBestPastIndividuals() # Still branching..
            #individual = self.poolman.getCertainIndividuals(["troubillion.json"])
            if not individual is None:
                for i in range(self.hp.mutatebestpool):
                    individual = self.custom_mutation(individual,0.01)[0]
                return self.creator.Individual([individual[0],individual[1],individual[2]])
        return self.creator.Individual([self.randomrewriter.randomstartstring(),self.randomrewriter.randomrules(),self.randomrewriter.setrandomconstants(self.hp)])

    def custom_crossover(self,ind1, ind2):
        """Custom crossover that crosses string1 with string1 and string2 with string2"""
        # Create copies to avoid modifying the original individuals
        new_ind1 = copy.deepcopy(ind1)
        new_ind2 = copy.deepcopy(ind2)
        # Crossover Start String
        if len(ind1[0]) > 1:  # Only cross if strings are long enough
            point = random.randint(1, len(ind1[0]) - 1)
            new_ind1[0] = ind1[0][:point] + ind2[0][point:]
            new_ind2[0] = ind2[0][:point] + ind1[0][point:]
        # Crossover Rules
        #smallerrules = min(len(new_ind1[1]),len(new_ind2[1]))
        #biggerrules = min(len(new_ind1[1]), len(new_ind2[1]))
        #choosenrulesindex = range(0,biggerrules)
        unchanged = OrderedDict()
        if len(new_ind1[1])>len(new_ind2[1]):
            biggerdict = new_ind1[1]
            smallerdict = new_ind2[1]
        else:
            biggerdict = new_ind2[1]
            smallerdict = new_ind1[1]

        for i in range(len(biggerdict)-len(smallerdict)):
            randomkey = random.choice(list(biggerdict.keys()))
            unchanged[randomkey] = biggerdict[randomkey]
            biggerdict.pop(randomkey)
        #if(len(biggerdict)==len(smallerdict)):
        #    raise Exception("A revoir..."+str(new_ind1)+str(new_ind2))
        index = 0
        crossed1,crossed2= OrderedDict(),OrderedDict()
        for key1, value1 in smallerdict.items():
            key2,value2 = list(biggerdict.items())[index]
            #print("key and value pairs",key1,key2,value1,value2)
            keys = self.crossstrings(key1,key2)
            values = self.crossstrings(value1,value2)
            crossed1[keys[0]] = values[0]
            crossed2[keys[1]] = values[1]
            index += 1
        # If some rules where left over
        if unchanged:
            for key,value in unchanged.items():
                ct = random.randint(0, 1)
                if ct :
                    crossed1[key] = value
                else:
                    crossed2[key] = value
        #print("crossed1",crossed1)
        #print("crossed2",crossed2)
        new_ind1[1] = crossed1
        new_ind2[1] = crossed2


        #print(new_ind1[2] , new_ind2[2] )

        keys1,keys2= list(new_ind1[2].keys()) , list(new_ind2[2].keys())
        resultdict1,resultdict2 = OrderedDict(),OrderedDict()
        similarkeys = list(set(keys1).intersection(keys2))

        for i in similarkeys:
            if i == 'center':
                realcenters = self.crossFloats(new_ind1[2][i][0], new_ind2[2][i][0])
                imagcenters = self.crossFloats(new_ind1[2][i][1], new_ind2[2][i][1])
                resultdict1[i] = [realcenters[0],imagcenters[1]]
                resultdict2[i] = [realcenters[1],imagcenters[0]]
            elif i == 'range':
                float1, float2 = self.crossFloats(new_ind1[2][i], new_ind2[2][i])
                resultdict1[i] = float1
                resultdict2[i] = float2
            # Here we have a float constant
            elif i.isupper() and len(i)==1:
                float1,float2 = self.crossFloats(new_ind1[2][i],new_ind2[2][i])
                resultdict1[i] = float1
                resultdict2[i] = float2
            # Here we have a fn
            elif i.islower() and len(i)==1:
                if not (new_ind1[2][i]['fntype'] == new_ind2[2][i]['fntype']):
                    # The Function type do not match, do nothing
                    # Overwrite one of the two fn
                    if random.random() < self.hp.crossingpblist[0]:
                        choosenfn = random.choice([new_ind1[2][i],new_ind2[2][i]])
                        resultdict1[i] = choosenfn
                        resultdict2[i] = choosenfn
                    else:
                        resultdict1[i] = new_ind1[2][i]
                        resultdict2[i] = new_ind2[2][i]

                elif new_ind1[2][i]['fntype'] == 'Taylorian' :
                    paramtensor1,paramtensor2 = new_ind1[2][i]['value'],new_ind2[2][i]['value']
                    maxcuttingpoint = min(paramtensor1.shape[0],paramtensor2.shape[0]) - 1
                    # If there is no cutting point, cat tensor as they are
                    if maxcuttingpoint == 0 :
                        resultdict1[i] = {'fntype': 'Taylorian', 'value' : torch.cat((paramtensor1,paramtensor2),dim=0)}
                        resultdict2[i]= {'fntype': 'Taylorian', 'value' : torch.cat((paramtensor2,paramtensor1),dim=0)}
                    # Here we cross both taylor terms tensors
                    else:
                        #print("maxcuttingpoint",maxcuttingpoint,paramtensor1,paramtensor2 )
                        cuttingpoint = random.randint(1,maxcuttingpoint)
                        resultdict1[i] = {'fntype': 'Taylorian',
                                          'value': torch.cat((paramtensor1[0:cuttingpoint],paramtensor2[cuttingpoint:]),dim=0)}
                        resultdict2[i] = {'fntype': 'Taylorian',
                                          'value': torch.cat((paramtensor2[0:cuttingpoint],paramtensor1[cuttingpoint:]),dim=0)}

                elif new_ind1[2][i]['fntype'] == 'Polynomial':
                    resultdict1[i] = new_ind1[2][i]
                    resultdict2[i] = new_ind2[2][i]
                    #print('before', resultdict1[i], resultdict2[i])
                    cutcoeff = random.random() < self.hp.crossingpblist[1]
                    cutexp = random.random() < self.hp.crossingpblist[2]
                    #print("cutcoeff,cutexp",cutcoeff,cutexp)
                    if cutcoeff:
                        paramtensor1, paramtensor2 = new_ind1[2][i]['value']['coefficients'], new_ind2[2][i]['value']['coefficients']
                        maxcuttingpoint = min(paramtensor1.shape[0], paramtensor2.shape[0]) - 1
                        # If there is no cutting point, cat tensor as they are
                        if maxcuttingpoint == 0:
                            resultdict1[i]['value']['coefficients'] = torch.cat((paramtensor1, paramtensor2), dim=0)
                            resultdict2[i]['value']['coefficients'] = torch.cat((paramtensor2, paramtensor1), dim=0)

                        # Here we cross both taylor terms tensors
                        else:
                            # print("maxcuttingpoint",maxcuttingpoint,paramtensor1,paramtensor2 )
                            cuttingpoint = random.randint(1, maxcuttingpoint)
                            resultdict1[i]['value']['coefficients'] = torch.cat(
                                                  (paramtensor1[0:cuttingpoint], paramtensor2[cuttingpoint:]), dim=0)
                            resultdict2[i]['value']['coefficients'] = torch.cat(
                                                  (paramtensor2[0:cuttingpoint], paramtensor1[cuttingpoint:]), dim=0)

                    if cutexp:
                        paramtensor1, paramtensor2 = new_ind1[2][i]['value']['exposants'], new_ind2[2][i]['value'][
                            'exposants']
                        maxcuttingpoint = min(paramtensor1.shape[0], paramtensor2.shape[0]) - 1
                        # If there is no cutting point, cat tensor as they are
                        if maxcuttingpoint == 0:
                            resultdict1[i]['value']['exposants'] = torch.cat((paramtensor1, paramtensor2), dim=0)
                            resultdict2[i]['value']['exposants'] = torch.cat((paramtensor2, paramtensor1), dim=0)
                        # Here we cross both taylor terms tensors
                        else:
                            # print("maxcuttingpoint",maxcuttingpoint,paramtensor1,paramtensor2 )
                            cuttingpoint = random.randint(1, maxcuttingpoint)
                            resultdict1[i]['value']['exposants'] = torch.cat(
                                (paramtensor1[0:cuttingpoint], paramtensor2[cuttingpoint:]), dim=0)
                            resultdict2[i]['value']['exposants'] = torch.cat(
                                (paramtensor2[0:cuttingpoint], paramtensor1[cuttingpoint:]), dim=0)

                elif new_ind1[2][i]['fntype'] == 'Rational':
                    resultdict1[i] = new_ind1[2][i]
                    resultdict2[i] = new_ind2[2][i]
                    #print('before', resultdict1[i], resultdict2[i])
                    cutcoeffnumerator = random.random() < self.hp.crossingpblist[3]
                    cutexpnumerator = random.random() < self.hp.crossingpblist[4]
                    cutcoeffdenominator = random.random() < self.hp.crossingpblist[5]
                    cutexpdenominator = random.random() < self.hp.crossingpblist[6]
                    #print("cutcoeff,cutexp",cutcoeff,cutexp)
                    if cutcoeffnumerator:
                        paramtensor1, paramtensor2 = new_ind1[2][i]['value']['numeratorcoefficients'], new_ind2[2][i]['value']['numeratorcoefficients']
                        maxcuttingpoint = min(paramtensor1.shape[0], paramtensor2.shape[0]) - 1
                        # If there is no cutting point, cat tensor as they are
                        if maxcuttingpoint == 0:
                            resultdict1[i]['value']['numeratorcoefficients'] = torch.cat((paramtensor1, paramtensor2), dim=0)
                            resultdict2[i]['value']['numeratorcoefficients'] = torch.cat((paramtensor2, paramtensor1), dim=0)

                        # Here we cross both taylor terms tensors
                        else:
                            # print("maxcuttingpoint",maxcuttingpoint,paramtensor1,paramtensor2 )
                            cuttingpoint = random.randint(1, maxcuttingpoint)
                            resultdict1[i]['value']['numeratorcoefficients'] = torch.cat(
                                                  (paramtensor1[0:cuttingpoint], paramtensor2[cuttingpoint:]), dim=0)
                            resultdict2[i]['value']['numeratorcoefficients'] = torch.cat(
                                                  (paramtensor2[0:cuttingpoint], paramtensor1[cuttingpoint:]), dim=0)

                    if cutexpnumerator:
                        paramtensor1, paramtensor2 = new_ind1[2][i]['value']['numeratorexposants'], new_ind2[2][i]['value'][
                            'numeratorexposants']
                        maxcuttingpoint = min(paramtensor1.shape[0], paramtensor2.shape[0]) - 1
                        # If there is no cutting point, cat tensor as they are
                        if maxcuttingpoint == 0:
                            resultdict1[i]['value']['numeratorexposants'] = torch.cat((paramtensor1, paramtensor2), dim=0)
                            resultdict2[i]['value']['numeratorexposants'] = torch.cat((paramtensor2, paramtensor1), dim=0)
                        # Here we cross both taylor terms tensors
                        else:
                            # print("maxcuttingpoint",maxcuttingpoint,paramtensor1,paramtensor2 )
                            cuttingpoint = random.randint(1, maxcuttingpoint)
                            resultdict1[i]['value']['numeratorexposants'] = torch.cat(
                                (paramtensor1[0:cuttingpoint], paramtensor2[cuttingpoint:]), dim=0)
                            resultdict2[i]['value']['numeratorexposants'] = torch.cat(
                                (paramtensor2[0:cuttingpoint], paramtensor1[cuttingpoint:]), dim=0)
                    if cutcoeffdenominator:
                        paramtensor1, paramtensor2 = new_ind1[2][i]['value']['denominatorcoefficients'], new_ind2[2][i]['value']['denominatorcoefficients']
                        maxcuttingpoint = min(paramtensor1.shape[0], paramtensor2.shape[0]) - 1
                        # If there is no cutting point, cat tensor as they are
                        if maxcuttingpoint == 0:
                            resultdict1[i]['value']['denominatorcoefficients'] = torch.cat((paramtensor1, paramtensor2), dim=0)
                            resultdict2[i]['value']['denominatorcoefficients'] = torch.cat((paramtensor2, paramtensor1), dim=0)

                        # Here we cross both taylor terms tensors
                        else:
                            # print("maxcuttingpoint",maxcuttingpoint,paramtensor1,paramtensor2 )
                            cuttingpoint = random.randint(1, maxcuttingpoint)
                            resultdict1[i]['value']['denominatorcoefficients'] = torch.cat(
                                                  (paramtensor1[0:cuttingpoint], paramtensor2[cuttingpoint:]), dim=0)
                            resultdict2[i]['value']['denominatorcoefficients'] = torch.cat(
                                                  (paramtensor2[0:cuttingpoint], paramtensor1[cuttingpoint:]), dim=0)

                    if cutexpdenominator:
                        paramtensor1, paramtensor2 = new_ind1[2][i]['value']['denominatorexposants'], new_ind2[2][i]['value'][
                            'denominatorexposants']
                        maxcuttingpoint = min(paramtensor1.shape[0], paramtensor2.shape[0]) - 1
                        # If there is no cutting point, cat tensor as they are
                        if maxcuttingpoint == 0:
                            resultdict1[i]['value']['denominatorexposants'] = torch.cat((paramtensor1, paramtensor2), dim=0)
                            resultdict2[i]['value']['denominatorexposants'] = torch.cat((paramtensor2, paramtensor1), dim=0)
                        # Here we cross both taylor terms tensors
                        else:
                            # print("maxcuttingpoint",maxcuttingpoint,paramtensor1,paramtensor2 )
                            cuttingpoint = random.randint(1, maxcuttingpoint)
                            resultdict1[i]['value']['denominatorexposants'] = torch.cat(
                                (paramtensor1[0:cuttingpoint], paramtensor2[cuttingpoint:]), dim=0)
                            resultdict2[i]['value']['denominatorexposants'] = torch.cat(
                                (paramtensor2[0:cuttingpoint], paramtensor1[cuttingpoint:]), dim=0)
                elif new_ind1[2][i]['fntype'] == 'Power':
                    resultdict1[i] = new_ind1[2][i]
                    resultdict2[i] = new_ind2[2][i]
                    iscrossing = random.random() < self.hp.crossingpblist[7]
                    if iscrossing:
                        float1,float2 = self.crossFloats(new_ind1[2][i]['value']['exponent'],new_ind2[2][i]['value']['exponent'])
                        resultdict1[i]['value']['exponent'] = float1
                        resultdict2[i]['value']['exponent'] = float2

                elif new_ind1[2][i]['fntype'] == 'Trigonometric':
                    resultdict1[i] = new_ind1[2][i]
                    resultdict2[i] = new_ind2[2][i]
                    iscrossing = random.random() < self.hp.crossingpblist[8]
                    if iscrossing:
                        resultdict1[i]['value']['trigooperation'] = new_ind2[2][i]['value']['trigooperation']
                        resultdict2[i]['value']['trigooperation'] = new_ind1[2][i]['value']['trigooperation']

                elif new_ind1[2][i]['fntype'] == 'Exponential':
                    resultdict1[i] = new_ind1[2][i]
                    resultdict2[i] = new_ind2[2][i]
                    iscrossing = random.random() < self.hp.crossingpblist[9]
                    if iscrossing:
                        float1,float2 = self.crossFloats(new_ind1[2][i]['value']['base'],
                                                                               new_ind2[2][i]['value']['base'])
                        resultdict1[i]['value']['base'] = float1
                        resultdict2[i]['value']['base'] = float2

                elif new_ind1[2][i]['fntype'] == 'Logarithmic':
                    resultdict1[i] = new_ind1[2][i]
                    resultdict2[i] = new_ind2[2][i]
                    iscrossing = random.random() < self.hp.crossingpblist[10]
                    if iscrossing:
                        float1, float2 = self.crossFloats(new_ind1[2][i]['value']['base'],
                                                          new_ind2[2][i]['value']['base'])
                        resultdict1[i]['value']['base'] = abs(float1)
                        resultdict2[i]['value']['base'] = abs(float2)

                        #print('after',resultdict1[i],resultdict2[i])
                #print("Crossover of a taylorian, result 1 :",resultdict1[i],"result 2 :",resultdict2[i])
        #print(resultdict1,resultdict2)

        differentkeys = list(set(keys1) - set(keys2))+list(set(keys2) - set(keys1))
        for i in differentkeys:
            if i in keys1:
                resultdict2[i] = new_ind1[2][i]
                resultdict1[i] = new_ind1[2][i]
            else:
                resultdict2[i] = new_ind2[2][i]
                resultdict1[i] = new_ind2[2][i]
        #print(resultdict1,resultdict2)
        new_ind1[2] = resultdict1
        new_ind2[2] = resultdict2


        #print(differentkeys)

        #
        # # Crossover second strings
        # if len(ind1[1]) > 1:  # Only cross if strings are long enough
        #     point = random.randint(1, len(ind1[1]) - 1)
        #     new_ind1[1] = ind1[1][:point] + ind2[1][point:]
        #     new_ind2[1] = ind2[1][:point] + ind1[1][point:]

        return new_ind1, new_ind2

    def crossFloats(self,float1, float2, eta=15):
        if random.random() <= 0.5:
            # Calculate beta value
            rand = random.random()
            if rand <= 0.5:
                beta = (2 * rand) ** (1.0 / (eta + 1))
            else:
                beta = (1.0 / (2.0 * (1.0 - rand))) ** (1.0 / (eta + 1))

            # Create two offspring
            float1 = 0.5 * ((1 + beta) * float1 + (1 - beta) * float2)
            float2 = 0.5 * ((1 - beta) * float1 + (1 + beta) * float2)

        return float1, float2

    def crossstrings(self,s1,s2):
        if len(s1) > 1 :  # Only cross if strings are long enough
            point = random.randint(1, len(s1) - 1)
            new_s1 = s1[:point] + s2[point:]
            new_s2 = s2[:point] + s1[point:]

        else:
            ct1,ct2,ct3,ct4,ct5,ct6  = random.randint(0,1),random.randint(0,1),random.randint(0,1),random.randint(0,1),random.randint(0,1),random.randint(0,1)
            new_s1 = s1 * ct1 + s2 * ct2
            new_s2 = s2 * ct3 + s1 * ct4
            if new_s1 == "":
                new_s1 = s1 * ct5 + s2 * abs(ct5-1)
            if new_s2 == "":
                new_s2 = s1 * ct6 + s2 * abs(ct6-1)
        return new_s1, new_s2

    def custom_mutation(self,individual, indpb):
        """Custom mutation that can modify characters in either string"""
        # Mutate first string
        for i in range(len(individual[0])):
            if random.random() < self.hp.mutationpblist[0]:
                individual[0] = (individual[0][:i] +
                                 random.choice(self.vocabulary + self.operators) +
                                 individual[0][i + 1:])
        rulesdict = individual[1]
        tempdict = OrderedDict()
        for i in list(rulesdict.keys()):
            newkey = i
            for j in range(len(i)):
                if random.random() < self.hp.mutationpblist[1]:
                    newkey = (newkey[:j] +
                                     random.choice(self.vocabulary + self.operators) +
                                     newkey[j + 1:])
            tempdict[newkey] = rulesdict[i]
        individual[1]=tempdict

        rulesdict = individual[1]
        tempdict = OrderedDict()
        for i in list(rulesdict.keys()):
            newvalue = rulesdict[i]
            for j in range(len(i)):
                if random.random() < self.hp.mutationpblist[2]:
                    newvalue = (newvalue[:j] +
                                     random.choice(self.vocabulary + self.operators) +
                                     newvalue[j + 1:])
            tempdict[i] = newvalue
        individual[1]=tempdict

        constantsdict = individual[2]
        for key,value in constantsdict.items():
            if key == "center":
                if random.random() < self.hp.mutationpblist[3]:
                    constantsdict[key][0] = constantsdict[key][0] * self.random_special(endian=False)
                if random.random() < self.hp.mutationpblist[4]:
                    constantsdict[key][0] = constantsdict[key][0] * self.random_special(endian=True)
                if random.random() < self.hp.mutationpblist[5]:
                    constantsdict[key][1] = constantsdict[key][1] * self.random_special(endian=False)
                if random.random() < self.hp.mutationpblist[6]:
                    constantsdict[key][1] = constantsdict[key][1] * self.random_special(endian=True)
                continue
            if key == "range":
                if random.random() < self.hp.mutationpblist[7]:
                    constantsdict[key] = constantsdict[key] * self.random_special(endian=False)
                if random.random() < self.hp.mutationpblist[8]:
                    constantsdict[key] = constantsdict[key] * self.random_special(endian=True)
                continue
            # Float constant
            if key.isupper():
                if random.random() < self.hp.mutationpblist[9]:
                    constantsdict[key] = complex(constantsdict[key].real * self.random_special(endian=False),constantsdict[key].imag)
                if random.random() < self.hp.mutationpblist[10]:
                    constantsdict[key] = complex(constantsdict[key].real * self.random_special(endian=True),constantsdict[key].imag)
                if random.random() < self.hp.mutationpblist[11]:
                    constantsdict[key] = complex(constantsdict[key].real,constantsdict[key].imag * self.random_special(endian=False))
                if random.random() < self.hp.mutationpblist[12]:
                    constantsdict[key] = complex(constantsdict[key].real,constantsdict[key].imag * self.random_special(endian=True))
            # Fn
            else:
                if constantsdict[key]['fntype'] == 'Taylorian':
                    firstmut, secondmut = random.random() < self.hp.mutationpblist[13],random.random() < self.hp.mutationpblist[14]
                    if firstmut or secondmut:
                        #print("Taylorian mutation before",constantsdict[key])
                        # Mutate a taylorian

                            tayloriantermscount = constantsdict[key]['value'].shape[0]
                            termtomodifyindex = random.randint(0,tayloriantermscount-1)
                            multiplier = random.choice([1+0j,0+1j,1+1j])
                            if firstmut:
                                constantsdict[key]['value'][termtomodifyindex] = constantsdict[key]['value'][termtomodifyindex] * (self.random_special(endian=False) * multiplier)
                            if secondmut:
                                constantsdict[key]['value'][termtomodifyindex] = constantsdict[key]['value'][termtomodifyindex] * (self.random_special(endian=True) * multiplier)
                elif constantsdict[key]['fntype'] == 'Polynomial':
                    mutatecoefficient, mutateexposant = random.random() < self.hp.mutationpblist[15],random.random() < self.hp.mutationpblist[16]
                    mutatenormalized = random.random() < self.hp.mutationpblist[17]
                    #print(constantsdict[key])
                    polycoeffcount = constantsdict[key]['value']['coefficients'].shape[0]
                    polyexposantcount = constantsdict[key]['value']['exposants'].shape[0]
                    # print("Before",constantsdict[key]['value']['coefficients'],constantsdict[key]['value']['exposants'])
                    if mutatecoefficient:
                        multiplier = random.choice([1 + 0j, 0 + 1j, 1 + 1j])
                        termtomodifyindex = random.randint(0, polycoeffcount - 1)
                        constantsdict[key]['value']['coefficients'][termtomodifyindex] = constantsdict[key]['value']['coefficients'][termtomodifyindex] * (self.random_special(endian=mutatenormalized) * multiplier)
                    if mutateexposant:
                        multiplier = random.choice([1 + 0j, 0 + 1j, 1 + 1j])
                        termtomodifyindex = random.randint(0, polyexposantcount - 1)
                        constantsdict[key]['value']['exposants'][termtomodifyindex] = constantsdict[key]['value']['exposants'][termtomodifyindex] * (
                                    self.random_special(endian=mutatenormalized) * multiplier)
                elif constantsdict[key]['fntype'] == 'Rational':
                    mutatenumeratorcoefficient, mutatenumeratorexposant, mutatedenominatorcoefficient, mutatedenominatorexposant = random.random() < self.hp.mutationpblist[18], random.random() < \
                                                        self.hp.mutationpblist[19],self.hp.mutationpblist[20],self.hp.mutationpblist[21]
                    mutatenormalizednumerator,mutatenormalizeddenominator = random.random() < self.hp.mutationpblist[22],random.random() < self.hp.mutationpblist[23]
                    # print(constantsdict[key])
                    numeratorcoeffcount = constantsdict[key]['value']['numeratorcoefficients'].shape[0]
                    numeratorexposantcount = constantsdict[key]['value']['numeratorexposants'].shape[0]
                    # print("Before",constantsdict[key]['value']['coefficients'],constantsdict[key]['value']['exposants'])
                    if mutatenumeratorcoefficient:
                        multiplier = random.choice([1 + 0j, 0 + 1j, 1 + 1j])
                        termtomodifyindex = random.randint(0, numeratorcoeffcount - 1)
                        constantsdict[key]['value']['numeratorcoefficients'][termtomodifyindex] = \
                        constantsdict[key]['value']['numeratorcoefficients'][termtomodifyindex] * (
                                    self.random_special(endian=mutatenormalizednumerator) * multiplier)
                    if mutatenumeratorexposant:
                        multiplier = random.choice([1 + 0j, 0 + 1j, 1 + 1j])
                        termtomodifyindex = random.randint(0, numeratorexposantcount - 1)
                        constantsdict[key]['value']['numeratorexposants'][termtomodifyindex] = \
                        constantsdict[key]['value']['numeratorexposants'][termtomodifyindex] * (
                                self.random_special(endian=mutatenormalizednumerator) * multiplier)
                    denominatorcoeffcount = constantsdict[key]['value']['denominatorcoefficients'].shape[0]
                    denominatorexposantcount = constantsdict[key]['value']['denominatorexposants'].shape[0]
                    if mutatedenominatorcoefficient:
                        multiplier = random.choice([1 + 0j, 0 + 1j, 1 + 1j])
                        termtomodifyindex = random.randint(0, denominatorcoeffcount - 1)
                        constantsdict[key]['value']['denominatorcoefficients'][termtomodifyindex] = \
                        constantsdict[key]['value']['denominatorcoefficients'][termtomodifyindex] * (
                                    self.random_special(endian=mutatenormalizeddenominator) * multiplier)
                    if mutatedenominatorexposant:
                        multiplier = random.choice([1 + 0j, 0 + 1j, 1 + 1j])
                        termtomodifyindex = random.randint(0, denominatorexposantcount - 1)
                        constantsdict[key]['value']['denominatorexposants'][termtomodifyindex] = \
                        constantsdict[key]['value']['denominatorexposants'][termtomodifyindex] * (
                                self.random_special(endian=mutatenormalizeddenominator) * multiplier)
                elif constantsdict[key]['fntype'] == 'Power':
                    firstmut, secondmut = random.random() < self.hp.mutationpblist[24],random.random() < self.hp.mutationpblist[25]

                    multiplier = random.choice([1 + 0j, 0 + 1j, 1 + 1j])
                    if firstmut:
                        constantsdict[key]['value']['exponent'] = constantsdict[key]['value']['exponent'] * (self.random_special(
                            endian=False) * multiplier)
                    if secondmut:
                        constantsdict[key]['value']['exponent'] = constantsdict[key]['value']['exponent'] * (
                                    self.random_special(
                                        endian=True) * multiplier)
                elif constantsdict[key]['fntype'] == 'Trigonometric':
                    constantsdict[key]['value']['trigooperation'] = random.choice(self.trigo.allowedoperations)

                elif constantsdict[key]['fntype'] == 'Exponential':
                    firstmut, secondmut = random.random() < self.hp.mutationpblist[26], random.random() < \
                                          self.hp.mutationpblist[27]

                    multiplier = random.choice([1 + 0j, 0 + 1j, 1 + 1j])
                    if firstmut:
                        constantsdict[key]['value']['base'] = constantsdict[key]['value']['base'] * (
                                    self.random_special(
                                        endian=False) * multiplier)
                    if secondmut:
                        constantsdict[key]['value']['base'] = constantsdict[key]['value']['base'] * (
                                self.random_special(
                                    endian=True) * multiplier)
                elif constantsdict[key]['fntype'] == 'Logarithmic':
                    firstmut, secondmut = random.random() < self.hp.mutationpblist[28], random.random() < \
                                          self.hp.mutationpblist[29]

                    multiplier = random.choice([1 + 0j, 0 + 1j, 1 + 1j])
                    if firstmut:
                        constantsdict[key]['value']['base'] = abs(constantsdict[key]['value']['base'] * (
                                self.random_special(
                                    endian=False) * multiplier))
                    if secondmut:
                        constantsdict[key]['value']['base'] = abs(constantsdict[key]['value']['base'] * (
                                self.random_special(
                                    endian=True) * multiplier))


        individual[2] = constantsdict

        return individual,
    def massiveMutation(self,population,mutationrate):
        mutated_population = []

        for individual in population:
            individual = self.custom_mutation(individual,mutationrate)
            mutated_population.append(individual)

        return mutated_population

    def random_special(self,endian=None):
        if endian is None:
            endian = random.random() < 0.5
        if endian:  # 50% chance for each range
            return random.random()*2-1  # Uniform between 0 and 1
        else:
            # For [1,∞), we can use inverse transform sampling
            # Using 1/x distribution from 1 to infinity
            return 1 / (1 - random.random()*2)
    def evaluate(self,individual):

        rewriter = RewriterEngine(individual[0],individual[1],individual[2],None,maxiter=100)
        starttime = time.time()

        #xi_penality
        if not rewriter.rewrite():
            #print("blou")
            return 400000000000,
        issane = rewriter.sanitycheck()
        rewriter.calculateConstants()
        if issane:

            #rewriter.runAllLines()
            self.rwevaluator.setRewriter(rewriter)
            #score = self.rwevaluator.evaluateByLineDifference()
            #rwevaluatorescore = self.rwevaluator.evaluateRewriter()
            #rwcompressionratio = self.rwevaluator.evaluateRewriterByCompressionRatio()
            #print("calculate sym",rewriter.getLinesAsString())


            # simpletrianglescore = self.rwevaluator.simpletrianglescore()
            # if simpletrianglescore < .6:
            #     return simpletrianglescore*1000000,

            # mintrianglescore = 0.1*0
            # trianglescore = self.rwevaluator.parabolictrianglescore()
            # if trianglescore < mintrianglescore:
            #     if trianglescore < 0:
            #         return 30000,
            #     else:
            #         return (1-trianglescore) * 10000 + 20000,
            #
            #symetryscore = self.rwc.mesureSymetry(rewriter.getLinesAsString())
            #
            # #print("rwevaluatorescore",rwevaluatorescore)
            # #print("rwcompressionratio",rwcompressionratio)
            # print("trianglescore",trianglescore)
            # print("symetryscore",symetryscore)
            #print(rewriter.getLinesAsString())
            # minsymetryscore = 0.1*0

            if not 'wavelet_variance' in vars(self):
                self.wavelet_variance = NormVar()
            if not 'wavelet_entropy' in vars(self):
                self.wavelet_entropy = NormVar()
            if not 'exitvalueentropy' in vars(self):
                self.exitvalueentropy = NormVar()
            if not 'bailoutentropy' in vars(self):
                self.bailoutentropy = NormVar()
            if not 'symetryscore' in vars(self):
                self.symetryscore = NormVar()
            if not 'timepenality' in vars(self):
                self.timepenality = NormVar()
            if True: #symetryscore >= minsymetryscore:# and rwevaluatorescore > 0.1:
                self.fractal.__init__(self.sieve,center=rewriter.getFractalCenter(),range=rewriter.getFractalRange(),resolution=(64,64))
                rewriter.fractal = self.fractal
                rewriter.runAllLines()
                if self.fractal.zeroBailouttest():
                    return 100000,
                center = individual[2]['center']
                range = individual[2]['range']
                xi_dist = strange_distance(abs(range), 2)
                #print(self.fractal.bailoutexitvalues)
                #self.fractal.printcomplexmatrix()
                #fractalexitvalues = self.fractal.getNormalizedExitValues()
                #fractal_output = self.fractal.bailout
                fractal_output = self.fractal.getNormalizedExitValues()
                #infoentropy = self.sieve.getimageinformationsnote(fractal_output)
                rawbailout = self.fractal.bailout
                self.exitvalueentropy.set(self.sieve.getimageinformationsnote(fractal_output))
                self.bailoutentropy.set(self.sieve.getimageinformationsnote(rawbailout))
                self.symetryscore.set(self.rwc.mesureSymetry(rewriter.getLinesAsString()))

                #test = torch.distributions.distribution.Distribution.entropy(fractal_output)
                #shannon_entropy_abs = -self.ic.shannon_entropy(rawbailout)
                #variance = self.ic.variance_std_metrics(fractal_output)['pixel_variance']
                #mutual_information = self.ic.mutual_information(fractal_output)
                #glcm_features = self.ic.glcm_features(fractal_output)
                wavelet_coefficient_stats = self.ic.wavelet_coefficient_stats(fractal_output)
                self.wavelet_variance.set(    wavelet_coefficient_stats['wavelet_ll_variance']
                                            + wavelet_coefficient_stats['wavelet_lh_variance']
                                            + wavelet_coefficient_stats['wavelet_hl_variance']
                                            + wavelet_coefficient_stats['wavelet_hh_variance'])
                self.wavelet_entropy.set(       wavelet_coefficient_stats['wavelet_entropy_lh']
                                              + wavelet_coefficient_stats['wavelet_entropy_hl']
                                              + wavelet_coefficient_stats['wavelet_entropy_hh'])
                # wavelet_coefficient_stats = -( # wavelet_coefficient_stats['wavelet_ll_variance']
                #                              #+ wavelet_coefficient_stats['wavelet_lh_variance']
                #                              #+ wavelet_coefficient_stats['wavelet_hl_variance']
                #                              #+ wavelet_coefficient_stats['wavelet_hh_variance']
                #                               wavelet_coefficient_stats['wavelet_entropy_lh']
                #                              + wavelet_coefficient_stats['wavelet_entropy_hl']
                #                              + wavelet_coefficient_stats['wavelet_entropy_hh'])
                #glcm_features =0
                #print("variance",variance)
                #print("glcm_features", glcm_features)
                #if  bailoutentropy is None or exitvalueentropy is None: #or infoentropy is None:  # or recon is None :
                #    return 10000,
                #if wavelet_coefficient_stats is None: #or glcm_features['glcm_energy'] is None or glcm_features['glcm_homogeneity'] is None :
                #    return 10000,
                #entropy = glcm_features['glcm_energy']*glcm_features['glcm_contrast']*glcm_features['glcm_energy']*exitvalueentropy + exitvalueentropy/2
                # entropy = (self.wavelet_entropy.get()+self.wavelet_variance.get()+self.bailoutentropy.get()*3
                #
                #            +self.exitvalueentropy.get()*2+xi_dist*2)
                time_elapsed = time.time() -starttime
                self.timepenality.set(time_elapsed)
                entropy = self.bailoutentropy.get()*3 - self.timepenality.get()*3
                print("self.bailoutentropy.get()*3 - self.timepenality.get()*3",self.bailoutentropy.get()*3, - self.timepenality.get()*2)
                if entropy < self.minentropy:
                    self.minentropy = entropy
                if entropy > self.maxentropy:
                    self.maxentropy = entropy
                if (self.maxentropy-self.minentropy) == 0 :
                    return 20000,
                entropy = (entropy - self.minentropy) / (self.maxentropy-self.minentropy)
                minimumentropy = 0.11

                print("entropy",entropy)
                # if entropy <= minimumentropy:
                #     print("Entropy low")
                #     return (minimumentropy-entropy)*10000,
                #symentropy = symetryscore * entropy * 60000
                return -entropy * 100000, #- symentropy,
            else:
                print("symetry to low",(minsymetryscore-symetryscore)*300000 )
                return (minsymetryscore-symetryscore)*300000 , #(2 - 2 * rwcompressionratio)


            #exit()
            #print("getLinesAsString",rewriter.getLinesAsString())
            #print("First line", rewriter.stringlist[0])
            #entropy = self.sieve.getimageinformationsnote(fractal_output)
            #recon = self.sieve.getimageautoencoderreconstructionerror(fractal_output)
            #if entropy is None: #or recon is None :
            #    return 300000000000,
            #print(torch.min(fractal_output),torch.max(fractal_output))
            if not self.sieve.isbailouttensorvalid(fractal_output):
                return 20000,

            #positiveclasses = ['equilibroid','Impactoid','interferoid','Divergent']
            positiveclasses = ['bacteroid','Hyperboid','Divergent','interferoid','equilibroid']
            #positiveclasses = [ 'equilibroid','Dentelle','Spiraloid']
            #positiveclasses = ['equilibroid']
            negativeclasses = ['Errone','Pointioioid']

            positiveloss,negativeloss = self.sieve.getimageautoencoderclasscorrespondance(fractal_output,positiveclasses,negativeclasses,softmaxtemperature=15)
            #classentropy = 1 - self.sieve.getimageautoencoderclassentropy(fractal_output)
            #print("weightedpositiveloss,weightednegativeloss,entropy",positiveloss.item(),negativeloss.item(),entropy)
            #minimumentropy = 0.13
            alpha = .1
            #note = -(positiveloss-negativeloss)- entropy
            #positiveentropy = torch.relu(torch.tensor([entropy-minimumentropy])).item()
            #note = -rwcompressionratio/5 - rwevaluatorescore/5- entropy

            #repetitive = self.rwevaluator.diversityHOG(self.fractal.getNormalizedBailout(),comparativecount=20)
            #print("repetitive", repetitive)
            # mostfrequentelementratio = self.fractal.moreFrequentBailout()
            # if mostfrequentelementratio > .25:
            #     mostfrequentelementratio = 10000
            # else:
            #     mostfrequentelementratio = 0
            # print("Most frequent element ratio ",mostfrequentelementratio)
            #note = - (1 - self.sieve.comparetootherimagenote(fractal_output))*10000#+repetitive*50
            #note = entropy * trianglescore - entropy* symetryscore - entropy
            #note = -repetitive*(1-mostfrequentelementratio) + trianglescore
            #note = -rwevaluatorescore/10  - entropy/2

            print("symentropy",symentropy)
            note = (-positiveloss+negativeloss)*15000 -1000 -entropy *5000 - symentropy - trianglescore * 2000 +15000 #+ mostfrequentelementratio +repetitive * 10000+15000
            print("entropy",entropy,'symetryscore',symetryscore)
            print("note", note )

            return note,
            #return entropynote*note,
        else:
            print("Rewriter is not sane, returning 200000000000")
            return 20000,

    def print_population(self,population, generation):
        # Sort population by fitness (assuming minimization)
        print(population)

        sorted_pop = sorted(population, key=lambda ind: ind.fitness.values[0])

        print(f"\nGeneration {generation} Report:")
        print("-" * 50)
        print(f"{'Individual':<30} {'Fitness':<10}")
        print("-" * 50)
        for i, ind in enumerate(sorted_pop):
            # Format individual and fitness based on your needs
            ind_str = str(ind)[:27] + "..." if len(str(ind)) > 30 else str(ind)
            print(f"{ind_str:<30} {ind.fitness.values[0]:<10.12f}")
        print("-" * 50)
    def printindividualcode(self,individual):
        rewriter = RewriterEngine(individual[0], individual[1], individual[2], self.fractal, maxiter=100,
                                  maxlinelenght=20000)
        rewriter.rewrite()
        print(rewriter.getLinesAsString())
    def draw_individual(self,individual,resolution = (894*3,int(865*3)),notxtandjson=False):

        rewriter = RewriterEngine(individual[0], individual[1], individual[2], self.fractal, maxiter=100,maxlinelenght=20000)
        rewriter.rewrite()
        issane = rewriter.sanitycheck()
        if issane:
            self.fractal.__init__(self.sieve,center=rewriter.getFractalCenter(),range=rewriter.getFractalRange(),resolution= resolution)
            rewriter.calculateConstants()
            rewriter.runAllLines()
            if not notxtandjson:
                with open(self.conf.generateTimestamp(extension=".txt"), 'w') as fout:
                    fout.write(rewriter.getLinesAsString())
                with open(self.conf.generateTimestamp(extension=".json"), 'w') as fout:
                    jsondump = [individual[0], individual[1], individual[2]]
                    # print(jsondump)
                    # json.dump(list, jsondump)
                    json_string = jsonpickle.encode(jsondump)
                    fout.write(json_string)
            self.fractal.drawFractal(saveonly=True)
    def draw_individual_with_variations(self,individual,keyslist):
        originalindividual = copy.deepcopy(individual)
        variations = [-2j,-1j,-0.5j,-0.1j,0.1j,.5j,1j,2j]

        for key in keyslist:
            if key in individual[2]:
                print(individual[2][key])
                numelements = individual[2][key].shape[0]
                for i in range(numelements):
                    for variation in variations:
                        individual[2][key][i] = originalindividual[2][key][i] + variation
                        self.draw_individual(individual,(512,512))
                        individual[2][key][i] = originalindividual[2][key][i]






    def remove_duplicates(self,population):
        # Create a dictionary with fitness as key (only keeps one individual per fitness)
        unique_fitness = {}
        for ind in population:
            fitness = int(ind.fitness.values[0]*10)/10
            if fitness not in unique_fitness:
                unique_fitness[fitness] = ind

        return list(unique_fitness.values())

    def detect_fitness_stagnation(self, current_best_fitness: float,improvement_threshold=0.01,max_generations_no_improvement = 5) -> bool:
        stagnationtminutes = 5
        if current_best_fitness < self.absolute_best_fitness - improvement_threshold:
            self.absolute_best_fitness = current_best_fitness
            self.lastimprovementtime = time.time()
            self.generations_without_improvement = 0
            return False
        self.generations_without_improvement += 1
        if time.time() - self.lastimprovementtime > 60 * stagnationtminutes:
            print("Stagnation after",stagnationtminutes,"minutes")
            return 2

        if self.generations_without_improvement >= max_generations_no_improvement:
            print("Stagnation after ", self.generations_without_improvement, "try.")
            return 2

        return  1


        # """
        # Detect stagnation based on fitness improvement over time.
        # """
        # if current_best_fitness < self.absolute_best_fitness - improvement_threshold:
        #     self.absolute_best_fitness = current_best_fitness
        #     self.generations_without_improvement = 0
        #     return False
        #
        # self.generations_without_improvement += 1
        # returnvalue = (self.generations_without_improvement >= max_generations_no_improvement) * 2
        # if returnvalue == 2 :
        #     print("Stagnation after ",self.generations_without_improvement,"try.")
        #     return 2
        # else:
        #     return 1

    def main(self,trial: optuna.Trial):


        self.hp.inittrialGAhp(trial)
        print("Optuna study started")
        self.vocabulary = self.hp.initvocabularyhp(trial)
        self.randomrewriter = RewriterRandomGenerator(self.vocabulary, self.operators)
        print("Trial vocabulary :",self.vocabulary)

        print("Hyperparameters :",self.hp)

        # Initialize the toolbox
        toolbox = base.Toolbox()

        # Register our generation and evaluation functions
        toolbox.register("individual", self.generate_individual
                         )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.evaluate)

        # Register our custom genetic operators
        toolbox.register("mate", self.custom_crossover)
        toolbox.register("mutate", self.custom_mutation, indpb=self.hp.mutationpb)
        toolbox.register("select", tools.selTournament, tournsize=self.hp.tournsize)

        # Create population
        population = toolbox.population(n=self.hp.pop_size)
        fitnesses = map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        # Statistics setup
        stats = tools.Statistics(lambda ind: ind.fitness.values)

        #return mysum /len(x)

        stats.register("max", max)
        # stats.register("min", np.min)
        # stats.register("max", np.max)
        # stats.register("avg", np.mean)
        # stats.register("std", np.std)

        logbook = tools.Logbook()
        logbook.header = ["gen", "evals"] + stats.fields
        # Run the evolution
        NGEN = 50000
        lastbestfitness = 0
        self.lastimprovementtime = time.time()
        lastdrawnnote = -10000000000
        for gen in range(NGEN):
            # if gen % 10 == 0:
            print("Generation :",gen)
            print("Hyperparameters :",self.hp)
            # Select the next generation individuals
            offspring = algorithms.varAnd(population, toolbox, cxpb=self.hp.crossingpb, mutpb=self.hp.mutationpb)

            # Evaluate the offspring
            fits = toolbox.map(toolbox.evaluate, offspring)
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit

            # Replace population
            population = toolbox.select(offspring + population, k=len(population))

            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
            print(population)

            # population = self.massiveMutation(population, .8)
            # fitnesses = map(toolbox.evaluate, population)
            # for ind, fit in zip(population, fitnesses):
            #    ind.fitness.values = fit
            # print("massive mutation",population)
            # Record statistics
            record = stats.compile(population)
            logbook.record(gen=gen, evals=len(offspring), **record)

            # Print report
            print(logbook.stream)
            self.print_population(population, gen)
            #print(population.ind)
            # for ind in population:
            #     fitness = ind.fitness.values[0]

            best_ind = tools.selBest(population, k=1)[0]
            print("best_ind",best_ind.fitness.values)
            stagnationstate = self.detect_fitness_stagnation(best_ind.fitness.values[0],improvement_threshold=50,max_generations_no_improvement=100)
            if stagnationstate == 2:
                print("Starting a new population after ",gen, "generations")
                if best_ind.fitness.values[0] < -70000:
                    self.draw_individual(best_ind, resolution=(512, 512))
                return best_ind.fitness.values[0]
            #print("lastdraw", best_ind.fitness.values[0]-lastdrawnnote)

            # force redraw at random in case the population regressed
            #if random.randint(0,100) == 1:
            #    stagnationstate = 0
            self.printindividualcode(best_ind)
            print('score',best_ind.fitness.values[0])
            # if stagnationstate == 0:
            #     if best_ind.fitness.values[0] > lastdrawnnote+100:
            #         self.draw_individual(best_ind,resolution=(512,512))
            #         #self.draw_individual_with_variations(best_ind, ['a', 'b'])
            #         lastdrawnnote = best_ind.fitness.values[0]
            lastbestfitness = best_ind.fitness.values[0]

            #self.remove_duplicates(population)
        if best_ind.fitness.values[0] < -70000:
            self.draw_individual(best_ind, resolution=(512, 512))
        return best_ind.fitness.values[0]

        # population, logbook = algorithms.eaSimple(population, toolbox,
        #                                           cxpb=0.7,  # crossover probability
        #                                           mutpb=0.2,  # mutation probability
        #                                           ngen=ngen,
        #                                           stats=stats,
        #                                           verbose=True)
        #
        # # Get best solution
        # best_ind = tools.selBest(population, k=1)[0]
        # return best_ind, best_ind.fitness.values[0]

if __name__ == "__main__":
    # target1 = "Hello World bilibloublou"
    # target2 = "Python GA blam bllaaamm"
    gen = GeneticAlgorithm()


    # ind1,ind2 = gen.generate_individual(),gen.generate_individual()
    # #print(ind1,ind2)
    # cr1,cr2 = gen.custom_crossover(ind1,ind2)
    # #print(cr1,cr2)
    # mut = gen.custom_mutation(cr1,.7)
    #print(mut)
    # optuna-dashboard <storage-url> --port 8081
    # gaoptuna optimise pour les debut de soupe primordiale
    # gabestpool optimise avec l utilisation d un pool
    currentstudyname = "single_fractal_timepen2"
    # study = optuna.create_study(study_name=currentstudyname, storage="<storage-url>",
    #                             sampler=optuna.samplers.NSGAIIISampler)
    # exit()
    studymanager = StudyManager(currentstudyname,gen.main)
    #studymanager.flushdummystudy()

    #studymanager.getbestkparameters(k=500)
    studymanager.starttrial(nostorage=False)

    #studymanager.runUsingBestValuesAtRandom(k=100)
    #optuna.delete_study(study_name=currentstudyname, storage="<storage-url>")
    #
    # study = optuna.create_study(study_name=currentstudyname, storage="<storage-url>",sampler=optuna.samplers.NSGAIISampler)
    #
    # exit()




    # for i in range(10000):
    #     gen = GeneticAlgorithm()
    #     gen.main()
    #     print("Starting new generation")
    #print(f"Target strings: '{target1}', '{target2}'")
    #print(f"Best solution: '{best_solution[0]}', '{best_solution[1]}'")
    #print(f"Total matches: {fitness}")
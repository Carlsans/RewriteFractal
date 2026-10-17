import glob
import random
import time

import jsonpickle
import torch
import yaml

from src.Configuration import Configuration


class PoolManager:
    def __init__(self,conf):
        self.conf = conf
        # absolute path to search all text files inside a specific folder
        path = self.conf.filepath + "*.json"
        self.files = glob.glob(path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def getCertainIndividuals(self,filename):
        filename = random.choice(filename)
        filename = self.conf.filepath + filename
        print("filename",filename)
        with open(filename, 'r') as fread:
            try:
                individual = jsonpickle.decode(fread.read())
            except:
                print("Bad json", filename)
                return None
            # print(individual)
            # TODO add 'fntype' and 'value' keys to taylorian so it work with older json version
            # Should work....
            for key, value in individual[2].items():
                print("key,value", key, value)
                if key.islower() and len(key) == 1:
                    if isinstance(value, dict):
                        individual[2][key]['value'] = value['value']
                    else:
                        individual[2][key] = {'fntype': 'Taylorian', 'value': value.to(self.device)}
                if torch.is_tensor(individual[2][key]):
                    individual[2][key] = individual[2][key].to(self.device)
                print(individual[2][key])

            return individual
    def getBestPastIndividuals(self):
        random.seed(time.time())
        pastindividualsfiles = random.choices(self.files,k=1)
        pastindividuals = []
        for file in pastindividualsfiles:
            with open(file, 'r') as fread:
                try:
                    individual = jsonpickle.decode(fread.read())
                except :
                    print("Bad json",file)
                    return None
                #print(individual)
                #TODO add 'fntype' and 'value' keys to taylorian so it work with older json version
                # Should work....
                for key,value in individual[2].items():
                    print("key,value",key,value)
                    if key.islower() and len(key) == 1:
                        if isinstance(value,dict):
                            individual[2][key]['value'] = value['value']
                        else:
                            individual[2][key] = {'fntype' : 'Taylorian', 'value' : value.to(self.device)}
                    if torch.is_tensor(individual[2][key]):
                        individual[2][key] = individual[2][key].to(self.device)
                    print(individual[2][key])

                return individual

                # jsondump = [individual[0], individual[1], individual[2]]
                # # print(jsondump)
                # # json.dump(list, jsondump)
                # json_string = jsonpickle.encode(jsondump)
                # fout.write(json_string)

#poolman = PoolManager()
#poolman.getBestPastIndividuals()
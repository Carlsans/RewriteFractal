class GAHyperParameters:
    def __init__(self):
        self.pop_size,self.mutationpb,self.crossingpb,self.tournsize,self.fetchbestpool = None,None,None,None,None
        self.mutationpblist = []
        self.mutationpblistlen = 30
        self.crossingpblist = []
        self.crossingpblistlen = 11
        self.pop_size_range = (10,200) # 500
        self.mutationpb_range = (0.001,0.8) # 0.05
        self.crossingpb_range = (0.1,1) # 0.8
        self.tournsize_range = (1,30) #3
        self.fetchbestpool_range = (.5,.9)
        self.fetchbestpoolmutate_range = (1,20)
        self.optionalvocabulary_range = (1,25)
        self.forcedvocabulary = "CZ"
        self.optionalvocabulary = "ABDEFGHIJKLMNOPQRSTUVWRXY"
        self.taylorian_range = (1,20)
        self.taylorians = "abdefghijklmnopqrstuvwrxy"
        self.polynomialcoeffmultiplier_range = (0.00001,100)
        self.polynomialexposant_range = (1,100)

    def __str__(self):
        returnstring = "Population size : " +str(self.pop_size) + " Mutation pb : " + str(self.mutationpb) + " Crossing pb : " + str(self.crossingpb) + " Tournament size : " + str(self.tournsize)

        return returnstring
    def inittrialGAhp(self,trial):
        self.pop_size = trial.suggest_int("pop_size", self.pop_size_range[0], self.pop_size_range[1])
        self.mutationpb = trial.suggest_float("mutationpb", self.mutationpb_range[0], self.mutationpb_range[1])
        self.mutationpblist = [trial.suggest_float("mutationpblist" + str(i), self.mutationpb_range[0], self.mutationpb_range[1]) for
            i in range(self.mutationpblistlen)]
        self.crossingpblist = [trial.suggest_float("crossingpblist" + str(i), self.crossingpb_range[0], self.crossingpb_range[1]) for
            i in range(self.crossingpblistlen)]
        # self.hp.mutationpb = [0.1,.2]
        self.crossingpb = trial.suggest_float("crossingpb", self.crossingpb_range[0], self.crossingpb_range[1])
        self.tournsize = trial.suggest_int("tournsize", self.tournsize_range[0], self.tournsize_range[1])
        self.fetchbestpool = trial.suggest_float("fetchbestpool", self.fetchbestpool_range[0],
                                                    self.fetchbestpool_range[1])
        self.mutatebestpool = trial.suggest_int("fetchbestpoolmutate", self.fetchbestpoolmutate_range[0], self.fetchbestpoolmutate_range[1])
        self.polynomialcoeffmultiplier = trial.suggest_float("polynomialcoeffmultiplier", self.polynomialcoeffmultiplier_range[0],
                                                    self.polynomialcoeffmultiplier_range[1])
        self.polynomialexposant = trial.suggest_int("polynomialexposant", self.polynomialexposant_range[0], self.polynomialexposant_range[1])

        self.rationalnumeratorcoeffmultiplier = trial.suggest_float("rationalnumeratorcoeffmultiplier", self.polynomialcoeffmultiplier_range[0],
                                                    self.polynomialcoeffmultiplier_range[1])
        self.rationalnumeratorexposant = trial.suggest_int("rationalnumeratorexposant", self.polynomialexposant_range[0], self.polynomialexposant_range[1])
        self.rationaldenominatorcoeffmultiplier = trial.suggest_float("rationaldenominatorcoeffmultiplier", self.polynomialcoeffmultiplier_range[0],
                                                    self.polynomialcoeffmultiplier_range[1])
        self.rationaldenominatorexposant = trial.suggest_int("rationaldenominatorexposant", self.polynomialexposant_range[0], self.polynomialexposant_range[1])


    def initvocabularyhp(self,trial):
        selectedoptionalvocablen = trial.suggest_int("optionalvocabulary", self.optionalvocabulary_range[0], self.optionalvocabulary_range[1])
        selectedtaylorianslen = trial.suggest_int("taylorian_range", self.taylorian_range[0], self.taylorian_range[1])
        vocabulary = self.forcedvocabulary + self.optionalvocabulary[0:selectedoptionalvocablen] + self.taylorians[0:selectedtaylorianslen]
        return vocabulary
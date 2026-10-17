import torch

from src.functions.Exponential import Exponential
from src.functions.Logarithmic import Logarithmic
from src.functions.Polynomial import Polynomial
from src.functions.Power import Power
from src.functions.Rational import Rational
from src.functions.TaylorSeries import TaylorSeries
from src.functions.Trigonometric import Trigonometric


class RewriterEngine:
    def __init__(self,startstring,rewriterules,constantsvalues,fractal,maxiter=10,maxlinelenght = 200):
        self.currentstring = startstring
        self.rewriterules = rewriterules
        self.constantsvalues = constantsvalues
        self.fractal = fractal
        self.maxiter = maxiter
        self.maxlinelenght = maxlinelenght
        self.stringlist = [startstring]
        self.reservedchar = "$#1234567890"
        self.currentjoiningoperation = "*"
        self.uselastoperationasbaseoperation = True
        self.tailor = TaylorSeries()
        self.polynomial = Polynomial()
        self.rational = Rational()
        self.power = Power()
        self.trigonometric = Trigonometric()
        self.exponential = Exponential()
        self.logarithmic = Logarithmic()
        if self.containreservedcharacter(startstring):
            raise Exception("Start string contain reserved character")

    def containreservedcharacter(self,string):

        for c in self.reservedchar:
            if c in string:
                #print("Contain reserved char",c)
                return 1
        return 0

    def rewrite(self):
        self.currentstring = self.currentstring.replace(' ','')
        for i in range(self.maxiter):
            #print("Line ", i+1, self.currentstring)
            j=0
            for key in self.rewriterules:
                #print(key)
                self.currentstring = self.currentstring.replace(key, '#'+ str(j)+'#')
                j+=1
            j = 0
            for key in self.rewriterules:
                #print(key)
                self.currentstring = self.currentstring.replace('#' + str(j) + '#',self.rewriterules[key])
                j += 1
            self.stringlist.append(self.currentstring )
            if len(self.currentstring) > self.maxlinelenght:
                #print("len(self.currentstring)",len(self.currentstring),"line index : ",len(self.stringlist))
                return False
        return True
        #print(self.stringlist)
            # for key in self.rewriterules:
            #     self.currentstring.replace(key,self.rewriterules[key])
            #
            #
            # self.stringlist.append(self.currentstring)
    def getFractalCenter(self):
        if not 'center' in self.constantsvalues:
            self.constantsvalues['center'] = [0,0]
        #print("Fractal Center", self.constantsvalues['center'])
        return self.constantsvalues['center']
    def getFractalRange(self):
        if not 'range' in self.constantsvalues:
            self.constantsvalues['range'] = 2
        #print("Fractal Range", self.constantsvalues['range'])
        return self.constantsvalues['range']
    def calculateConstants(self):
        self.constantlist = []
        self.formatedlineslist = []
        def islast(column,string):
            return column == (len(string) - 1)


        # Iter l ensemble des lignes
        for i in self.stringlist:
            constantlistline = []
            # Check the first character of the line to set running constant value
            # Be sure it's not a taylorian
            if i[0] in self.constantsvalues and i[0].isupper() :
                runningconstant = self.constantsvalues[i[0]]
            else:
                runningconstant = 0
            currentline = ""
            formatedline = ""
            formatedparameterindex=0
            # Iter chaque caractere
            jindex = 0
            for j in i:
                # Here, the actual character is a constant
                if j in self.constantsvalues:
                    #currentline += str(self.constantsvalues[j])
                    # There is a next character
                    if not islast(jindex,i):
                        # Next character is Z or C and we have a taylorian so no need to calculate constants here
                        if i[jindex + 1] in "ZC" and i[jindex].islower():
                            currentline += j
                            formatedline += j
                        else:
                            # Next character is a constant
                            if i[jindex+1] in self.constantsvalues:
                                #print(i[jindex],i[jindex+1],jindex,runningconstant)
                                # No precedent constant value
                                if runningconstant == 0:
                                    # Here we have a fn and the next character is not another fn
                                    if i[jindex].islower() and not i[jindex + 1].islower():
                                        # Here we have a taylorian (legacy)
                                        if not isinstance(self.constantsvalues[i[jindex]] , dict) or self.constantsvalues[i[jindex]]['fntype']=='Taylorian':
                                            if not isinstance(self.constantsvalues[i[jindex]] , dict):
                                                self.tailor.setcoefficients(self.constantsvalues[i[jindex]])
                                            else:
                                                self.tailor.setcoefficients(self.constantsvalues[i[jindex]]['value'])
                                            runningconstant = self.tailor.getY(self.constantsvalues[i[jindex + 1]]).item()
                                        elif self.constantsvalues[i[jindex]]['fntype']=='Polynomial':
                                            self.polynomial.setvalues(self.constantsvalues[i[jindex]]['value'])
                                            runningconstant = self.polynomial.getY(self.constantsvalues[i[jindex + 1]])
                                        elif self.constantsvalues[i[jindex]]['fntype']=='Rational':
                                            self.rational.setvalues(self.constantsvalues[i[jindex]]['value'])
                                            runningconstant = self.rational.getY(self.constantsvalues[i[jindex + 1]])
                                        elif self.constantsvalues[i[jindex]]['fntype'] == 'Power':
                                            self.power.setvalues(self.constantsvalues[i[jindex]]['value'])
                                            runningconstant = self.power.getY(self.constantsvalues[i[jindex + 1]])
                                        elif self.constantsvalues[i[jindex]]['fntype'] == 'Trigonometric':
                                            self.trigonometric.setvalues(self.constantsvalues[i[jindex]]['value'])
                                            runningconstant = self.trigonometric.getY(self.constantsvalues[i[jindex + 1]])
                                        elif self.constantsvalues[i[jindex]]['fntype'] == 'Exponential':
                                            self.exponential.setvalues(self.constantsvalues[i[jindex]]['value'])
                                            runningconstant = self.exponential.getY(
                                                self.constantsvalues[i[jindex + 1]])
                                        elif self.constantsvalues[i[jindex]]['fntype'] == 'Logarithmic':
                                            self.logarithmic.setvalues(self.constantsvalues[i[jindex]]['value'])
                                            runningconstant = self.logarithmic.getY(
                                                self.constantsvalues[i[jindex + 1]])
                                            # Here the next character is a constant, not a taylorian
                                    elif not i[jindex + 1].islower():
                                        runningconstant = self.constantsvalues[i[jindex]] * self.constantsvalues[i[jindex + 1]] # Note, ici on devrait utiliser la derniere operation et non *
                                    # Here we have a first Constant followed by a taylorian
                                    elif i[jindex].isupper():
                                        runningconstant = self.constantsvalues[i[jindex]]
                                    # If we got 2 taylor fn we do nothing...
                                # A constant is being calculated
                                else:
                                    # We have a fn and the next character is not another fn
                                    # but we have a running constant
                                    if i[jindex].islower() and not i[jindex + 1].islower():
                                        if not isinstance(self.constantsvalues[i[jindex]] , dict) or self.constantsvalues[i[jindex]]['fntype']=='Taylorian':
                                            if not isinstance(self.constantsvalues[i[jindex]] , dict):
                                                self.tailor.setcoefficients(self.constantsvalues[i[jindex]])
                                            else:
                                                self.tailor.setcoefficients(self.constantsvalues[i[jindex]]['value'])
                                            runningconstant = runningconstant  * self.tailor.getY(self.constantsvalues[i[jindex + 1]]).item()
                                        elif self.constantsvalues[i[jindex]]['fntype']=='Polynomial':
                                            self.polynomial.setvalues(self.constantsvalues[i[jindex]]['value'])
                                            runningconstant = runningconstant  * self.polynomial.getY(self.constantsvalues[i[jindex + 1]])
                                        elif self.constantsvalues[i[jindex]]['fntype'] == 'Rational':
                                            self.rational.setvalues(self.constantsvalues[i[jindex]]['value'])
                                            runningconstant = runningconstant  * self.rational.getY(self.constantsvalues[i[jindex + 1]])
                                        elif self.constantsvalues[i[jindex]]['fntype'] == 'Power':
                                            self.power.setvalues(self.constantsvalues[i[jindex]]['value'])
                                            runningconstant = runningconstant  * self.power.getY(self.constantsvalues[i[jindex + 1]])
                                        elif self.constantsvalues[i[jindex]]['fntype'] == 'Trigonometric':
                                            self.trigonometric.setvalues(self.constantsvalues[i[jindex]]['value'])
                                            runningconstant = runningconstant  * self.trigonometric.getY(
                                                self.constantsvalues[i[jindex + 1]])
                                        elif self.constantsvalues[i[jindex]]['fntype'] == 'Exponential':
                                            self.exponential.setvalues(self.constantsvalues[i[jindex]]['value'])
                                            runningconstant = runningconstant  * self.exponential.getY(
                                                self.constantsvalues[i[jindex + 1]])
                                        elif self.constantsvalues[i[jindex]]['fntype'] == 'Logarithmic':
                                            self.logarithmic.setvalues(self.constantsvalues[i[jindex]]['value'])
                                            runningconstant = runningconstant  * self.logarithmic.getY(
                                                self.constantsvalues[i[jindex + 1]])
                                            ##self.tailor.setcoefficients(self.constantsvalues[i[jindex]])
                                        ##runningconstant = runningconstant  * self.tailor.getY(self.constantsvalues[i[jindex + 1]]).item()
                                    # Here the next character is a constant, not a fn
                                    elif not i[jindex + 1].islower():
                                        runningconstant = runningconstant  * self.constantsvalues[i[jindex+1]]
                                    # If we got 2 taylor fn we do nothing...

                            # Next character is not a constant
                            else:
                                # This constant is not associated with another constant
                                # And it's not a taylorian
                                # so append it's value in the constantlistline list directly
                                if runningconstant == 0 and i[jindex] in self.constantsvalues and i[jindex].isupper():
                                    currentline += str(self.constantsvalues[i[jindex]])
                                    formatedline += "$"+str(formatedparameterindex)+"$"
                                    formatedparameterindex += 1
                                    constantlistline.append(self.constantsvalues[i[jindex]])
                                # The constant was associated with other constant(s).
                                # Since the next character is not a constant, append the running constant value
                                # to the constantlistline list
                                else:
                                    currentline += str(runningconstant)
                                    formatedline += "$" + str(formatedparameterindex) + "$"
                                    formatedparameterindex += 1
                                    constantlistline.append(runningconstant)
                                runningconstant = 0
                    # This is the last character
                    else:
                        # This constant is associated with another so take the running constant
                        # and add it to constantlistline
                        if not runningconstant == 0:
                            currentline += str(runningconstant)
                            formatedline += "$" + str(formatedparameterindex) + "$"
                            formatedparameterindex += 1
                            constantlistline.append(runningconstant)
                        # If this is a constant and this constant is not associated with another
                        # And it is not a taylorian
                        # just plug it's value in constantlistline
                        elif i[jindex] in self.constantsvalues and i[jindex].isupper():
                            currentline += str(self.constantsvalues[i[jindex]])
                            formatedline += "$" + str(formatedparameterindex) + "$"
                            constantlistline.append(self.constantsvalues[i[jindex]])
                # Here we have something else, not a constant. We just report it as it is
                else:
                    # This is not a constant because it was not defined so
                    currentline += j
                    formatedline += j
                jindex += 1
            # Append constants from this line to constantlist
            self.constantlist.append(constantlistline)
            # Append this formated line to formatedlineslist
            self.formatedlineslist.append(formatedline)
            #print(currentline)
            #print(constantlistline)
            #print(formatedline)
        #print(self.constantlist)
        #print(self.formatedlineslist)

    def runFormatedLine(self,formatedline,constants):
        #print(formatedline,constants)
        readingconstant = ""
        baseoperation = "*"
        lastoperation = baseoperation
        currentoperation = None
        currentconstant = None
        currentvalue = None
        def applyoperation(operation,op1,op2):
            if operation == '+':
                return op1 + op2
            if operation == '-':
                return op1 - op2
            if operation == '*':
                return op1 * op2
            if operation == '/':
                if not torch.is_tensor(op2) and op2 == 0 + 0j:
                    op2 = 0.000000000001+0.0000000000001j
                return op1 / op2
            # That's a fn
            if operation.islower():
                if operation in self.constantsvalues:
                    if self.constantsvalues[operation]['fntype'] == 'Taylorian':
                        self.tailor.setcoefficients(self.constantsvalues[operation]['value'])
                        return applyoperation(lastoperation, op1, self.tailor.getY(op2))
                    elif self.constantsvalues[operation]['fntype'] == 'Polynomial':
                        self.polynomial.setvalues(self.constantsvalues[operation]['value'])
                        return applyoperation(lastoperation, op1, self.polynomial.getY(op2))
                    elif self.constantsvalues[operation]['fntype'] == 'Rational':
                        self.rational.setvalues(self.constantsvalues[operation]['value'])
                        return applyoperation(lastoperation, op1, self.rational.getY(op2))
                    elif self.constantsvalues[operation]['fntype'] == 'Power':
                        self.power.setvalues(self.constantsvalues[operation]['value'])
                        return applyoperation(lastoperation, op1, self.power.getY(op2))
                    elif self.constantsvalues[operation]['fntype'] == 'Trigonometric':
                        self.trigonometric.setvalues(self.constantsvalues[operation]['value'])
                        return applyoperation(lastoperation, op1, self.trigonometric.getY(op2))
                    elif self.constantsvalues[operation]['fntype'] == 'Exponential':
                        self.exponential.setvalues(self.constantsvalues[operation]['value'])
                        return applyoperation(lastoperation, op1, self.exponential.getY(op2))
                    elif self.constantsvalues[operation]['fntype'] == 'Logarithmic':
                        self.logarithmic.setvalues(self.constantsvalues[operation]['value'])
                        return applyoperation(lastoperation, op1, self.logarithmic.getY(op2))
                else:
                    return op2
                # Here we use * has default operation todo use last operation
                # newoperation = None
                # if newoperation is None:
                #     newoperation = '*'
                #print(lastoperation,op1,"sum op 2",torch.sum(op2))
                #print(lastoperation,"op1",op1,"taylor",torch.sum(self.tailor.getY(op2)))
                #####  return applyoperation(lastoperation,op1,self.tailor.getY(op2))

            raise Exception("Wrong operation type : "+str(operation))

        for char in formatedline:
            # Read a constant parameter
            # Readind last $ sign and get the current constant
            if char == "$" and len(readingconstant) > 0:
                #print(readingconstant)
                currentconstant = constants[int(readingconstant.lstrip('$'))]
                readingconstant = ""
                if currentvalue is None:
                    currentvalue = currentconstant
                else:
                    if currentoperation is None:
                        if self.uselastoperationasbaseoperation:
                            currentvalue = applyoperation(lastoperation, currentvalue, currentconstant)
                        else:
                            currentvalue = applyoperation(baseoperation, currentvalue, currentconstant)
                    else:
                        currentvalue = applyoperation(currentoperation, currentvalue, currentconstant)
                        currentoperation = None

            # Reading first $ sign
            elif char == "$" and len(readingconstant) == 0:
                readingconstant += char
            # Here we add numbers that correspond the index
            elif len(readingconstant) > 0:
                readingconstant += char
            # Here we got an operation or a variable
            else:
                if char == "C":
                    if currentvalue is None:
                        currentvalue = self.fractal.C
                        if not currentoperation is None:
                            if currentoperation == "-":
                                currentvalue = - currentvalue
                    else:
                        if currentoperation is None:
                            if self.uselastoperationasbaseoperation:
                                currentvalue = applyoperation(lastoperation,currentvalue,self.fractal.C)
                            else:
                                currentvalue = applyoperation(baseoperation,currentvalue,self.fractal.C)
                        else:
                            currentvalue = applyoperation(currentoperation, currentvalue, self.fractal.C)
                            currentoperation = None

                if char == "Z":
                    #print("z")
                    if currentvalue is None:
                        currentvalue = self.fractal.Z
                        if not currentoperation is None:
                            if currentoperation == "-":
                                currentvalue = - currentvalue
                    else:
                        if currentoperation is None:
                            if self.uselastoperationasbaseoperation:
                                currentvalue = applyoperation(lastoperation, currentvalue, self.fractal.Z)
                            else:
                                currentvalue = applyoperation(baseoperation, currentvalue, self.fractal.Z)
                        else:
                            currentvalue = applyoperation(currentoperation, currentvalue, self.fractal.Z)
                            currentoperation = None
                # Here we got an operation or a taylorian
                if char in "+-*/" or char.islower():
                    #print("operator",char)
                    currentoperation = char
                    if char in "+-*/":
                        lastoperation = char





        return currentvalue
    def runAllLines(self):
        for i in range(len(self.formatedlineslist)):
            self.fractal.Z = self.runFormatedLine(self.formatedlineslist[i], self.constantlist[i])
            #print(i)
            if i == 0 and self.fractal.invalidZtest():
                #print("invalid z = 0")
                return None
            if i == 5 and self.fractal.zeroBailouttest():
                #print("invalid bailout")
                return None

            #print(self.fractal.Z)
            result = self.fractal.bailouttest(i+1)
            # Something wrong here
            if not result:
                break
        self.fractal.addnotbailedout()

    def getLinesAsString(self):

        return "\n".join(self.stringlist)


    def sanitycheck(self):
        maxoperationbyline = 20
        maxfunctionsbyline = 40
        lineindex = 0
        for line in self.stringlist:
            functionsoccurencesbyline = sum(1 for char in line if char.islower())
            if functionsoccurencesbyline > maxfunctionsbyline :
                print(functionsoccurencesbyline, "functions occurences on line", lineindex)
                print("Sanity check failed")
                return False
            if line.count('Z') + line.count('C') > maxoperationbyline:
                print(line.count('Z') + line.count('C'),"occurence on line",lineindex)
                print("Sanity check failed")
                return False
            lineindex += 1
        return True

# # rewriter = RewriterEngine("Z * +IZ + C / -I", rewriterules)
# #startstring = "A+A+Z*BCA"
# rewriterules = {'A':'BD','B':'AD','D':'Ba'}
# constantsvalues = {'A':3,'B':2,'D':1,'a':torch.tensor([0,1,0,0,0,0])}
# constantsvalues['b'] = torch.tensor([1,0,-1,0,1])
# constantsvalues['t'] = torch.tensor([0,1,0,2,0,4])
#
# #startstring = "Z * +IZ/A +C"
# #rewriterules =  {'+I':'-I','-I':'+I','-C':'+C','+C':'-C','A':'AB','BB':'BA','BA':'C','AAA':'','BBB':'','AA':'B','BB':'A'}
# #constantsvalues = {'I' : 1,'A':.03452+.3453j,'B':.406633+.04553j}
#
# #line = rewriter.runFormatedLine(rewriter.formatedlineslist[0],rewriter.constantlist[0])
# #print(line)
# sieve = Sieve()
# fractal = Fractal(sieve)
# #rewriter = RewriterEngine("ZaCBAZaBAC", rewriterules,constantsvalues,fractal)
# rewriter = RewriterEngine("aZ*aZ+C", rewriterules,constantsvalues,fractal,maxiter=100)
# rewriter.rewrite()
# print("rewriter.getLinesAsString()",rewriter.getLinesAsString())
# rewriter.calculateConstants()
# print("rewriter.constantlist",rewriter.constantlist)
# print("rewriter.formatedlineslist",rewriter.formatedlineslist)
# rewriter.runAllLines()
# print(rewriter.fractal.drawFractal())
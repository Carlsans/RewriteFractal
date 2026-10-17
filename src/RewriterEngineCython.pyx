# cython: boundscheck=False, wraparound=False, nonecheck=False

#import numpy as np
#cimport numpy as np
cimport cython
from libc.math cimport pow

class RewriterEngineCython:

    def fib(self,n):
        """Print the Fibonacci series up to n."""
        a, b = 0, 1
        while b < n:
            print(b, end=' ')
            a, b = b, a + b

        print()
    def factorialadd(self,n):
        cdef:
            long f = 0
            int i = 0
        while i<= n:
            f += i
            i += 1
        return f
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)
    @cython.nonecheck(False)
    def mesureSymetry(self,mystring):

        if not mystring[-1] == "\n":
            mystring += "\n"
        # index = 0
        # endofline = len(mystring)
        # line = ""
        # totalscore = 0
        # scorebyline=0
        # linecount = 0
        cdef:
            Py_ssize_t index = 0
            Py_ssize_t endofline = len(mystring)
            Py_ssize_t linecount = 0
            Py_ssize_t linelenght, maxdistance
            double totalscore = 0.0
            double scorebyline = 0.0
            int linenorm
            str line = ""
            str firstpart, secondpart
            set subset
            Py_ssize_t i, j, t
        while index < endofline:
            if mystring[index] == '\n':
                #print("line", line)
                linecount +=1
                linelenght = len(line)
                for i in range(1,linelenght):
                    maxdistance = min(i,linelenght-i)
                    #print("maxdistance",maxdistance)
                    for j in range(maxdistance):

                        firstpart= line[i-j-1:i][::-1]
                        secondpart = line[i:i+j+1]
                        #print(firstpart,"vs",secondpart)
                        if firstpart==secondpart:
                            subset = set(firstpart)

                            #("subset",subset)
                            #print("win",pow(len(firstpart),len(subset)))
                            scorebyline += pow(len(firstpart),len(subset)-1)-1
                            #scorebyline += pow(len(subset),len(subset))
                    #print(line[i],end='')
                #print()
                #print("line",line,"scorebyline",scorebyline)
                if linelenght > 1:
                    if linelenght % 2 == 0:
                        linenorm = self.factorialadd(linelenght/2)+self.factorialadd((linelenght-2)/2)
                    else:
                        linenorm = 0


                    totalscore += scorebyline / linenorm
                scorebyline = 0
                line = ""
            line += mystring[index]
            index += 1
        totalscore = totalscore / linecount if linecount > 0 else 0.0
        return totalscore
        #print("totalscore",totalscore,"linecount",linecount)

        # c_line: string = u""
        # ustring: unicode = mystring
        # for uchar in ustring:
        #     c_line.append(uchar)
        #     #print(c_line)
        #     if uchar == u'\n':
        #         ...
                #print("Found the letter A")
    # def mesureSymetry(self, input_string: str) -> None:
    #     c_line: string = string()  # Initialize an empty C++ string
    #     char: cython.char
    #     # Iterate through the characters in the string
    #     for char in input_string.encode('utf-8'):  # UTF-8 encoded bytes
    #         if char == b'\n'[0]:  # Check for newline character
    #             break
    #         c_line += chr(char)  # Append character to C++ string
    #
    #     # Use `c_line` as needed
    #     print(f"C++ string: {c_line}")
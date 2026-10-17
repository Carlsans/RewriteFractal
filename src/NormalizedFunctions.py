from math import exp


def strange_distance(x,n):
    if x <= n:
        return  (x/n)
    else:
        return   exp(-(x-n))


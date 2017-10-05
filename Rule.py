import math

class Rule:
    def __init__(self,c,s,p,q):
        self.c = c
        self.s = s
        self.p = p
        self.q = q

    def pertinencia(self, x):
        return math.exp(-0.5*((x-self.c)**2 / self.s**2))

    def y(self, x):
        return self.p * x + self.q

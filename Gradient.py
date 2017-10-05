import random as rnd
import FuzzyModel
import numpy as np
class Gradient:

    def __init__(self, x, y, modelo):
        self.x = x
        self.y = y
        self.modelo = modelo

        rand_ind = rnd.randint(0, len(x) - 1)
        self.xi = x[rand_ind]
        self.yi = y[rand_ind]
        self.m = len(self.modelo.regras)

        self.ys = self.modelo.yEstimado(self.x)
        self.yk = np.array([self.modelo.regras[i].y(self.xi) for i in range(self.m)])
        self.yk[np.isnan(self.yk)] = 0
        self.wk = np.array([self.modelo.regras[i].pertinencia(self.xi) for i in range(self.m)])
        self.wk[np.isnan(self.wk)] = 0
        self.wks = sum(self.wk)

        self.eq1 = -2*(self.yi-self.ys)
        self.eq2 = [self.wk[i]/self.wks for i in range(self.m)]
        self.eq3 = [(self.yk[i]-self.ys)/self.wks for i in range(self.m)]
        self.eq4 = [self.wks/self.wk[i] for i in range(self.m)]
        self.eq5 = [(self.xi - self.modelo.regras[i].c)/self.wks for i in range(self.m)]

    def derivadaParcialC(self):
        dJdC = np.zeros(self.m)
        for k in range(self.m):
            equation = (self.wk[k] * (self.x - self.modelo.regras[k].c) / self.modelo.regras[k].s ** 2)

            if np.isnan(self.eq1) or np.isnan(self.eq3[k]) or np.isnan(self.eq4[k]) or np.isnan(
                    self.eq5[k]):
                raise Exception()
            dJdC[k] = self.eq1 * self.eq3[k] * self.eq4[k] * self.eq5[k] * equation

        return dJdC

    def derivadaParcialS(self):
        dJdS = np.zeros(self.m)
        for k in range(self.m):
            equation = (self.wk[k] * ((self.xi - self.modelo.regras[k].c) ** 2) / self.modelo.regras[k].s ** 3)
            dJdS[k] = self.eq1 * self.eq3[k] * self.eq4[k] * self.eq5[k] * equation
        return dJdS

    def derivadaParcialP(self):
        dJdp = [self.eq1 * self.eq2[k] * self.xi for k in range(self.m)]
        return dJdp

    def derivadaParcialQ(self):
        dJdq = [self.eq1 * self.eq2[k] for k in range(self.m)]
        return dJdq


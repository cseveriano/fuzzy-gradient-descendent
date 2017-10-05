import Rule
import numpy as np

class FuzzyModel:
    def __init__(self, m, c, s, p, q):
        self.regras = []
        self.m = m

        for k in range(self.m):
            self.regras.append(Rule(c[k],s[k],p[k],q[k]))

    def yEstimado(self, xi):
        self.m = len(self.regras)
        w = np.array([self.regras[k].pertinencia(xi) for k in range(self.m)])
        ys = np.array([self.regras[k].y(xi) for k in range(self.m)])

        return sum([w[k] * ys[k] for k in range(self.m)]) / sum(w)

    def rmse(self, x, y):
        return 1

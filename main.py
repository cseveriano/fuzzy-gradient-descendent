import random as rnd
import numpy as np
import math
import FuzzyModel
import Gradient

def sinc(xx):
    return math.sin(xx)/xx

def initRandomParams(x, m):
    c = np.array([rnd.uniform(min(x), max(x)) for k in range(m)])
    s = np.array([rnd.uniform(0.0, 1.0) for k in range(m)])
    p = np.array([rnd.uniform(-1.0, 1.0) for k in range(m)])
    q = np.array([rnd.uniform(-1.0, 1.0) for k in range(m)])

    ret = [c, s, p, q]
    return ret


x = np.arange(0,2*math.pi,0.01)
y = [sinc(xx) for xx in x]
m = 5

c, s, p, q = initRandomParams(x,m)

modelo = FuzzyModel(m, c, s, p, q)

melhor_modelo = modelo

rmse = modelo.rmse(x, y)

count = 0
min_erro = 0.0001
max_iter = 100
alpha = 0.001
melhor_rmse = rmse

while rmse > min_erro and count <= max_iter:
    grad = Gradient(x, y, modelo)

    dJdc = grad.derivadaParcialC()
    dJds = grad.derivadaParcialS()
    dJdp = grad.derivadaParcialP()
    dJdq = grad.derivadaParcialQ()



    for k in range(m):
        c[k] = c[k] - alpha * dJdc[k]
        s[k] = s[k] - alpha * dJds[k]
        p[k] = p[k] - alpha * dJdp[k]
        q[k] = q[k] - alpha * dJdq[k]

        modelo = FuzzyModel(m, c, s, p, q)

        rmse = modelo.rmse(x, y)

        if rmse < melhor_rmse:
            melhor_mse = rmse
            melhor_modelo = modelo

        count += 1


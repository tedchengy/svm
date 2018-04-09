#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: chen
"""
import cvxpy
import numpy as np
import matplotlib.pyplot as plt
# Problem data



q1x = np.loadtxt('q1x.dat')
q1y = np.loadtxt('q1y.dat')

X = q1x;
y = 2*(q1y-0.5);
C = 1;
m = np.size(q1x,0);
n = np.size(q1x,1);

w = cvxpy.Variable(n)
#w = cvxpy.Variable((n,1))
b = cvxpy.Variable()
xi = cvxpy.Variable(m)

#objective = cvxpy.Minimize(1/2*cvxpy.sum_entries(np.multiply(w,w))+C*cvxpy.sum_entries(xi))
objective = cvxpy.Minimize(1/2*cvxpy.sum_entries(cvxpy.norm(w,2)**2)+C*cvxpy.sum_entries(xi))
constraints = [cvxpy.mul_elemwise(y,X*w+b)>= 1 - xi, xi>=0]
#constraints = [cvxpy.norm(y,X*w+b)**2>= 1 - xi, xi>=0]
prob = cvxpy.Problem(objective, constraints)
result = prob.solve()
print('result=',result)
print('w=',w.value)
print('b=',b.value)

xp = np.linspace(min(X[:,0]), max(X[:,0]), 100)
yp = - (w.value[0]*xp + b.value)/w.value[1]
yp1 = - (w.value[0]*xp + b.value-1)/w.value[1]
yp0 = - (w.value[0]*xp + b.value+1)/w.value[1]
idx0 =[i for i in range(len(q1y)) if q1y[i]==0]
idx1 =[i for i in range(len(q1y)) if q1y[i]==1]

tiltle=('decision boundary for a linear SVM classifier with C='+str(C))
plt.title(tiltle)
plt.plot(q1x[idx0, 0], q1x[idx0, 1],'rx')
plt.plot(q1x[idx1, 0], q1x[idx1, 1], 'go');
plt.plot(xp, np.array(yp)[0], '-b', xp,  np.array(yp1)[0], '--g', xp,  np.array(yp0)[0], '--r');
#plt.hold('off') 
plt.show()


#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def f(x):
    return 2*np.exp(-x/2)*np.cos(8*x) + 0.8*np.exp(-x/10)*np.cos(2*x)

def f_fit(x,p):
    return p[0]*np.exp(-x/p[1])*np.cos(p[2]*x) + p[3]*np.exp(-x/p[4])*np.cos(p[5]*x)

#Pusedo measured data by random number
xdata = np.linspace(-1, 10, num=10000)
np.random.seed(1234)
ydata = f(xdata) + 0.1*np.random.randn(xdata.size)

#Least squares method with scipy.optimize
def fit_func(parameter,x,y):
    residual = y-(f_fit(x,parameter))
    return residual

parameter0 = [3,1,7,0.9,9,3]
result = optimize.leastsq(fit_func,parameter0,args=(xdata,ydata))
print(result)
fit_par = np.zeros(len(parameter0), dtype =  float)
fit_par = result[0]
print(fit_par)

#PLot
plt.figure(figsize=(8,5))
plt.plot(xdata,ydata,'bo', label='Exp.')
plt.plot(xdata,f_fit(xdata,fit_par),'k-',label='fitted line', linewidth=10, alpha=0.3)
plt.plot(xdata,f(xdata),'r-', label='mother line', linewidth=3)
plt.xlabel('Time')
plt.ylabel('Verocity')
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)
plt.show()

#%%

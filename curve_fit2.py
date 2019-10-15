#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
   
   # ===========================================================
   # フィッティングのクラス
   # ===========================================================
class FITTING(object):
    # -------------------------------------------------------
    # コンストラクター
    #       # -------------------------------------------------------
    def __init__(self, function, x, y):
        self.f = function
        self.x = x
        self.y = y

   
       # -------------------------------------------------------
       # フィッティングの実行
       # -------------------------------------------------------
     def do_fitting(self):
       self.popt, self.pcov = curve_fit(self.f, self.x, self.y)
   
       print('a: {0:e}\nb: {1:e}\nc: {2:e}'. format(self.popt[0], self.popt[1], self.popt[2]))
   
   
       # -------------------------------------------------------
       # フィッティング結果のプロット
       # -------------------------------------------------------
       def plot(self, Nx=65):
           xmin, xmax = min(self.x), max(self.x)
           xp = np.linspace(xmin, xmax, Nx)
           fig = plt.figure()
           plot = fig.add_subplot(1,1,1)
           plot.set_xlabel("x", fontsize=12, fontname='serif')
           plot.set_ylabel("y", fontsize=12, fontname='serif')
           plot.tick_params(axis='both', length=10, which='major')
           plot.tick_params(axis='both', length=5,  which='minor')
           plot.set_xlim(xmin, xmax)
           plot.set_ylim([-1.2,1.2])
           plot.minorticks_on()
           plot.plot(xp, self.f(xp, *self.popt), 'b-')
           plot.plot(x, y, 'ro', markersize=10)
           fig.tight_layout()
           plt.show()
           #fig.savefig('result.pdf', orientation='portrait', transparent=False, bbox_inches=None, frameon=None)
           fig.clf()
   
   
   # -------------------------------------------------------
   # データをフィッティングする関数
   # -------------------------------------------------------
 def fit_func(x, a, b, c):
       return a*np.sin(x)*np.exp(-b*x)+c
       
   
   # -------------------------------------------------------
   # メイン関数
   # -------------------------------------------------------
   if __name__ == "__main__":
   
       x = np.linspace(0, 6*np.pi, 32)
       noise = 0.05*np.random.normal(size=x.size)
       y = np.sin(x)*np.exp(-x/5) + noise
   
       fit = FITTING(fit_func, x, y)
       fit.do_fitting()
       fit.plot(Nx=257)
       

#%%

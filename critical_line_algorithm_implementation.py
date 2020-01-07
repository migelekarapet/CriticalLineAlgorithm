import matplotlib.pyplot as plt
import numpy as np  
import matplotlib.pyplot as plt  
import cvxopt as opt  
from cvxopt import blas, solvers  
import pandas as pd

np.random.seed(345)

# Turn off progress printing  
solvers.options['show_progress'] = False  

## NUMBER OF STOCKS  
number_stocks = 4
 
number_observ = 1000

result_array = np.random.randn(number_stocks, number_observ)::: {.highlight .hl-ipython2}  

plt.plot(result_array.T, alpha=.4);  
plt.xlabel('time')  
plt.ylabel('results')  


def arbitr_mass(n):  
        k = np.random.rand(n)  
        return k / sum(k)

print arbitr_mass(number_stocks)  
print arbitr_mass(number_stocks)  

def random_portfolio(results):  
    p = np.asmatrix(np.mean(results, axis=1))  
    w = np.asmatrix(arbitr_mass(results.shape[0]))  
    C = np.asmatrix(np.cov(results))  
    mu = w * p.T  
    sigma = np.sqrt(w * C * w.T)  
    # This recursion reduces outliers to keep plots pretty  
    if sigma > 2:  
        return random_portfolio(results)  
    return mu, sigma  

n_portfolios = 500  
means, stds = np.column_stack([  
    random_portfolio(result_array)  
    for _ in xrange(n_portfolios)  
])

plt.plot(stds, means, 'o', markersize=5)  
plt.xlabel('sigma')  
plt.ylabel('expected value')  
plt.title('The expected value and sigma for the pottfolio returns')  

def optimal_portfolio(results):  
    n = len(results)  
    results = np.asmatrix(results)  
    N = 100  
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]  
     
    S = opt.matrix(np.cov(results))  
    pbar = opt.matrix(np.mean(results, axis=1))  
   
    G = -opt.matrix(np.eye(n))  
    h = opt.matrix(0.0, (n ,1))  
    A = opt.matrix(1.0, (1, n))  
    b = opt.matrix(1.0)  
    
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x']  
                  for mu in mus]  
   
    results = [blas.dot(pbar, x) for x in portfolios]  
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]  
    
    m1 = np.polyfit(results, risks, 2)  
    x1 = np.sqrt(m1[2] / m1[0])  
    
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']  
    return np.asarray(wt), results, risks

weights, results, risks = optimal_portfolio(result_array)

plt.plot(stds, means, 'o')  
plt.ylabel('mean')  
plt.xlabel('std')  
plt.plot(risks, results, 'y-o')  

print weights  


### backtesting on real data

from zipline.utils.factory import load_bars_from_yahoo  
end = pd.Timestamp.utcnow()  
start = end - 2500 * pd.tseries.offsets.BDay()

data = load_bars_from_yahoo(stocks=['AAPL', 'AMZN', 'IBM', 'MSFT',  
                                    'HPQ', 'FB', 'JNJ'],  
                            start=start, end=end)  
data.loc[:, :, 'price'].plot(figsize=(8,5))  
plt.ylabel('price in $')  


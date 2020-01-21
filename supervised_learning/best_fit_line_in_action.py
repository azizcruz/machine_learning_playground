from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style


xs = np.array([1,2,3,4,5,6,7], dtype=np.float64)
ys = np.array([6,3,5,8,9,2,11], dtype=np.float64)

def best_fit_slop_intercept(xs, ys):
    m = ( (mean(xs) * mean(ys) ) - mean(xs*ys) ) / ( (mean(xs)**2) - mean(xs**2) )
    b = mean(ys) - m*mean(xs)

    return m, b


m, b = best_fit_slop_intercept(xs, ys)

regression_line = [(m*x) + b for x in xs]

print(regression_line)

predict_x = 8
predict_y = (m*predict_x) + b

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y)
plt.plot(xs, regression_line)
plt.show()
from sklearn.datasets import load_iris
import numpy as np
iris = load_iris()
X = iris.data  # Matriz de características
y = iris.target  # Vector de etiquetas
print("Dimensiones de X:", X.shape)
print("Dimensiones de y:", y.shape)
n_dim = X.shape[1]
n_data = X.shape[0]
x_format = X
y_format = y - 1
y1 = np.zeros(n_data)
y2 = np.zeros(n_data)
y3 = np.zeros(n_data)
for i in range(n_dim):
    x_format[:,i] = X[:,i]/np.max(np.abs(X[:,i]))
for i in range(n_data):
    if y_format[i] != -1:
        y1[i] = 0
    else:
        y1[i] = 1

    if y_format[i] != 0:
        y2[i] = 0
    else:
        y2[i] = 1

    if y_format[i] != 1:
        y3[i] = 0
    else:
        y3[i] = 1



from common.logger import create_logger
from common.evaluations import *
from common.plots import *
from core.schmidt import Istar
from core.data_analysis import *
import matplotlib.pyplot as plt

log = create_logger()

base_functions = 350
r_data = 1/n_data
data_area = r_data*np.ones((n_data,n_dim))

my_istar = Istar(logger = log, data_dimensions = n_dim, base_functions = base_functions)
# my_istar.init_weights(method = "common", val = 100)
# my_istar.init_bias(method = "lineal")

# (bias,weights) = distribute_bases(x_format, base_functions)
# my_istar.init_weights(method = "external", val = weights)
# my_istar.init_bias(method = "external", val = bias)


# (bias,weights,data_area) = analyze_data(x_format, base_functions, 0.08)
(bias,weights,data_area) = analyze_data2(inputs = x_format, n_base_funcs = base_functions, w_coef = 0.1, n_anchors = 30)
my_istar.init_weights(method = "external", val = weights)
my_istar.init_bias(method = "external", val = bias)

# data_area = evaluate_influence_areas(x_format)

my_istar.evaluate_ortogonal_base()
b_values = evaluate_base(weights, bias, x_format)
o_values = evaluate_orts(b_values, my_istar.betas)
dibujar_vectores(range(n_data),o_values,"Ortogonal bases",False)

my_istar.evaluate_proyection(x_format, data_area, y_format)
output = my_istar.run(x_format)

# my_istar.evaluate_proyection(x_format, data_area, y1)
# output1 = my_istar.run(x_format)

# my_istar.evaluate_proyection(x_format, data_area, y2)
# output2 = my_istar.run(x_format)

# my_istar.evaluate_proyection(x_format, data_area, y3)
# output3 = my_istar.run(x_format)

# output = -output1 + output2*0 + output3

error = np.mean(np.abs(y_format - output))*100
print(error)
plt.figure(figsize=(8, 6))
plt.plot(y_format, label="Original", color='blue')
plt.plot(output, label="Predicho", color='red')
plt.grid(True)
plt.legend()
plt.show()

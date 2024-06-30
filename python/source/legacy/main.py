import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
from source.plots import dibujar_vectores
from source.evaluations import evaluate_base,evaluate_orts

print("---START---\n")
### Define base functions

print("Evaluating base functions")
n_dim = 1 # Data dimensions
n_base_funcs = 25 # First layer neurons
n_points_per_dimension = int(np.power(n_base_funcs,1/n_dim))
n_base_funcs = np.power(n_points_per_dimension,n_dim)

# Linear initialization
base_funcs_weights = np.empty(n_base_funcs, dtype=object)
base_funcs_bias = np.empty(n_base_funcs, dtype=object)

bias_step = 2/n_points_per_dimension
weights = 100*np.ones(n_dim)
bias = -1*np.ones(n_dim)+bias_step/2
dim_cnt = np.zeros(n_dim)

base_funcs_weights[0] = copy.copy(weights)
base_funcs_bias[0] = copy.copy(bias)
for i in range(1,n_base_funcs):
    base_funcs_weights[i] = copy.copy(weights)

    j = 0
    while True:
        dim_cnt[j] += 1
        bias[j] += bias_step
        if dim_cnt[j] == n_points_per_dimension:
            dim_cnt[j] = 0
            bias[j] = -1
            j += 1
        else:
            break
    base_funcs_bias[i] = copy.copy(bias)

## Beta matrix
zero_tolerance =1e-8
# Proyections and norms for exp(x)
proyections_b2b = np.ones((n_base_funcs,n_base_funcs))
norms_b = np.zeros(n_base_funcs)
for i in tqdm(range(n_base_funcs), desc="Base proyections", ncols=80, ascii=True):
    for j in range(i, n_base_funcs):
        for d in range(n_dim):
            b1 = base_funcs_bias[i][d]
            b2 = base_funcs_bias[j][d]
            w1 = base_funcs_weights[i][d]
            w2 = base_funcs_weights[j][d]
            dim_coef = 0

            if b1==b2 and w1==w2:
                integration_sections = 1
                integration_limits = np.zeros(3)
                integration_abs_signs = np.ones(2)
                integration_limits[0] = -1
                integration_limits[1] = 1

                if b1 > integration_limits[0]:
                    integration_abs_signs[0] = -1

                if b1 > integration_limits[0] and b1 < integration_limits[1]:
                    integration_sections = 2
                    integration_limits[2] = integration_limits[1]
                    integration_limits[1] = b1

                for s in range(integration_sections):
                    low_lim = integration_limits[s]
                    up_lim = integration_limits[s+1]

                    w1s = integration_abs_signs[s]*w1
                    dim_coef += -1/(w1s*(1+w1s*(up_lim - b1))) + 1/(w1s*(1+w1s*(low_lim - b1)))
            else:
                integration_sections = 1
                integration_limits = np.zeros(4)
                integration_abs_signs = np.ones((2,3))
                integration_limits[0] = -1

                lims = np.sort([b1,b2])
                arg_lims = np.argsort([b1,b2])
                if lims[0] <= -1:
                    if lims[1] <= -1:
                        integration_limits[1] = 1
                        integration_abs_signs[0,0] = 1
                        integration_abs_signs[1,0] = 1
                    elif lims[1] >= 1:
                        integration_limits[1] = 1
                        integration_abs_signs[arg_lims[0],0] = 1
                        integration_abs_signs[arg_lims[1],0] = -1
                    else:
                        integration_sections = 2

                        integration_limits[1] = lims[1]
                        integration_abs_signs[arg_lims[0],0] = 1
                        integration_abs_signs[arg_lims[1],0] = -1

                        integration_limits[2] = 1
                        integration_abs_signs[arg_lims[0],1] = 1
                        integration_abs_signs[arg_lims[1],1] = 1
                elif lims[0] >= 1:
                    integration_limits[1] = 1
                    integration_abs_signs[0,0] = -1
                    integration_abs_signs[1,0] = -1
                elif lims[1] >= 1:
                    integration_sections = 2

                    integration_limits[1] = lims[0]
                    integration_abs_signs[arg_lims[0],0] = -1
                    integration_abs_signs[arg_lims[1],0] = -1

                    integration_limits[2] = 1
                    integration_abs_signs[arg_lims[0],1] = 1
                    integration_abs_signs[arg_lims[1],1] = -1
                else:
                    integration_sections = 3

                    integration_limits[1] = lims[0]
                    integration_abs_signs[arg_lims[0],0] = -1
                    integration_abs_signs[arg_lims[1],0] = -1

                    integration_limits[2] = lims[1]
                    integration_abs_signs[arg_lims[0],1] = 1
                    integration_abs_signs[arg_lims[1],1] = -1

                    integration_limits[3] = 1
                    integration_abs_signs[arg_lims[0],2] = 1
                    integration_abs_signs[arg_lims[1],2] = 1

                for s in range(integration_sections):
                    low_lim = integration_limits[s]
                    up_lim = integration_limits[s+1]

                    w1s = integration_abs_signs[0,s]*w1
                    w2s = integration_abs_signs[1,s]*w2

                    den = -1 + w1s/w2s - w1s*b2 + w1s*b1
                    if den == 0:
                        print("Zero in Proyection of i: "+str(i)+" over j: "+str(j))
                        print(" W1s = "+str(w1s))
                        print(" W2s = "+str(w2s))
                        print(" b1 = "+str(b1))
                        print(" b2 = "+str(b2))
                        exit()
                    num2 = 1/den
                    num1 = w1s/w2s*num2

                    dim_coef += num1*(1/w1s*np.log(np.abs(1+w1s*(up_lim - b1))) - 1/w1s*np.log(np.abs(1+w1s*(low_lim - b1))))
                    dim_coef -= num2*(1/w2s*np.log(np.abs(1+w2s*(up_lim - b2))) - 1/w2s*np.log(np.abs(1+w2s*(low_lim - b2))))

            proyections_b2b[i,j] *=dim_coef
        proyections_b2b[j,i] = proyections_b2b[i,j]
    norms_b[i] = np.sqrt(proyections_b2b[i,i])

if(np.isnan(proyections_b2b).any()):
    print("B2B PROYECTIONS: NAN ERROR")
    print(proyections_b2b)
    exit()

# Betas
betas = np.zeros((n_base_funcs,n_base_funcs))
proyections_o2b = np.zeros((n_base_funcs,n_base_funcs))
norms_o = np.zeros(n_base_funcs)

norms_o[0] = norms_b[0]
betas[0,0] = 1/norms_b[0]
for i in tqdm(range(1, n_base_funcs), desc="Betas Matrix", ncols=80, ascii=True):
    norms_o[i] = np.pow(norms_b[i],2)
    for j in range(i):
        proyections_o2b[i,j] = proyections_b2b[i,j]
        for k in range(j):
            proyections_o2b[i,j] -= proyections_o2b[j,k]*proyections_o2b[i,k]
        proyections_o2b[i,j] /= norms_o[j]
        norms_o[i] -= np.pow(proyections_o2b[i,j],2)
    norms_o[i] = np.sqrt(norms_o[i])
    if(np.isnan(norms_o[i]).any()):
        plt.figure(figsize=(8, 6))
        plt.plot(norms_b[0:i], label='norms_b', color='red')
        plt.plot(norms_o[0:i], label='norms_o', color='blue')
        plt.grid(True)
        plt.legend()
        plt.show()
        print("NAN ERROR")
        print(i)
        exit()

    betas[i,i] = 1/norms_o[i]
    for j in range(i):
        for k in range(j, i):
            betas[i,j] -= proyections_o2b[i,k]*betas[k,j]
        betas[i,j] /= norms_o[i]

if(np.isnan(proyections_o2b).any()):
    print("O2B PROYECTIONS: NAN ERROR")
    print(proyections_o2b)
    exit()

if(np.isnan(betas).any()):
    print("BETAS: NAN ERROR")
    print(betas)
    exit()

### Evaluate f

print("\nEvaluating data function")
## Define DATA
n_data = 500
r_data = 1/n_data

# Random initialization
data_in = np.empty(n_data, dtype=object)
data_lab = np.zeros(n_data)
for i in range(n_data):
    data_in[i] = 2*np.random.rand(n_dim) - 1
    data_lab[i] = -np.ones(n_dim)# np.sin(np.sum(data_in[i]))

# Proyections
proyections_f2b = np.zeros(n_base_funcs)
proyections_f2o = np.zeros(n_base_funcs)
for i in tqdm(range(n_base_funcs), desc="Function proyections", ncols=80, ascii=True):
    for d in range(n_data):
        data_sum = 1
        for dim in range(n_dim):
            integration_sections = 1
            integration_limits = np.zeros(3)
            integration_abs_signs = np.ones(2)
            integration_limits[0] = np.max([data_in[d][dim] - r_data, -1])
            integration_limits[1] = np.min([data_in[d][dim] + r_data, 1])

            b1 = base_funcs_bias[i][dim]
            if b1 > integration_limits[0]:
                integration_abs_signs[0] = -1

            if b1 > integration_limits[0] and b1 < integration_limits[1]:
                integration_sections = 2
                integration_limits[2] = integration_limits[1]
                integration_limits[1] = b1

                integration_abs_signs[0] = -1
                integration_abs_signs[1] = 1

            w1 = base_funcs_weights[i][dim]
            dim_coef = 0
            for s in range(integration_sections):
                low_lim = integration_limits[s]
                up_lim = integration_limits[s+1]

                w1s = integration_abs_signs[s]*w1
                dim_coef += 1/w1s*np.log(np.abs(1+w1s*(up_lim - b1))) - 1/w1s*np.log(np.abs(1+w1s*(low_lim - b1)))
            data_sum *=dim_coef
        proyections_f2b[i] += data_lab[d]*data_sum

if(np.isnan(proyections_f2b).any()):
    print("F2B PROYECTIONS: NAN ERROR")
    print(proyections_f2b)
    exit()

for i in range(0, n_base_funcs):
    for j in range(i, -1, -1):
        proyections_f2o[i] += betas[i,j]*proyections_f2b[j]

if(np.isnan(proyections_f2o).any()):
    print("F2O PROYECTIONS: NAN ERROR")
    print(proyections_f2o)
    exit()

# Sigmas
sigma = np.zeros(n_base_funcs)
for i in range(n_base_funcs):
    for j in range(n_base_funcs):
        sigma[i] += proyections_f2o[j]*betas[j,i]

if(np.isnan(sigma).any()):
    print("SIGMAS: NAN ERROR")
    print(sigma)
    exit()

### Test error

print("\nTesting")
network_output = np.zeros(n_data)
error = 0
rel_error = 0
for d in range(n_data):
    first_layer_output = np.zeros(n_base_funcs)
    for n in range(n_base_funcs):
        first_layer_output[n] = np.prod(1/(1+np.abs(base_funcs_weights[n]*(data_in[d] - base_funcs_bias[n]))))

    network_output[d] = np.sum(first_layer_output*sigma)
    error += np.abs(data_lab[d] - network_output[d])
    rel_error += np.abs((data_lab[d] - network_output[d])/data_lab[d])

error /= n_data
rel_error *= 100/n_data

print("     Test results:")
print("         Average Absolute error = " + str(error))
print("         Average Relative error = " + str(rel_error))


n_vals = 5000
base_values = np.empty(n_base_funcs, dtype=object)
ort_values = np.empty(n_base_funcs, dtype=object)
for b in range(n_base_funcs):
    base_values[b] = evaluate_base(base_funcs_weights[b],base_funcs_bias[b],n_vals)
for b in range(n_base_funcs):
    ort_values[b] = evaluate_orts(base_values,betas,b)
for i in range(n_base_funcs-1):
    for j in range(i,n_base_funcs):
        # bas_val = np.sum(base_values[i]*base_values[j])*2/n_vals
        # print("Proyection i:"+str(i)+" over j: "+str(j)+"   = "+str(bas_val))
        ort_val = np.sum(ort_values[i]*ort_values[j])*2/n_vals
        print("Proyection i:"+str(i)+" over j: "+str(j)+"   = "+str(ort_val))

dibujar_vectores(np.linspace(-1,1,n_vals),base_values,"Base vectors",False)
dibujar_vectores(np.linspace(-1,1,n_vals),ort_values,"Ortogonal Vectors",True)



# # Generar 100 valores de x entre -1 y 1
# x = np.linspace(-1, 1, 100)

# # Definir la función f(x) = exp(-|x|)
# y = np.exp(-np.abs(x))

# # Crear la gráfica
# plt.figure(figsize=(8, 6))
# plt.plot(x, y, label=r'$f(x) = \exp(-|x|)$', color='blue')
# plt.title(r'Gráfica de $f(x) = \exp(-|x|)$')
# plt.xlabel('x')
# plt.ylabel(r'$f(x)$')
# plt.grid(True)
# plt.legend()
# plt.show()

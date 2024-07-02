import numpy as np
import copy


class Istar:

    def __init__(self, logger, data_dimensions, base_functions):
        """
        Istar is the core class for the multidimensional Gram-Schidt procedure.

        """

        self.logger = logger

        self.n_dim = data_dimensions
        self.n_base_funcs = base_functions

    def init_weights(self, method, val):
        """
        Initialize the first layer weights.
        There are several initialization ways supported.
            common      -> Same value for all weights
            random      -> Random values within a defined range
            external    -> Specify values

        :param method: Initialization method
        :param val: common      -> Initialization value
                    random      -> Random initialization range
                    external    -> Weights matrix

        """

        self.weights = np.ones((self.n_base_funcs,self.n_dim))
        if method == "common":
            if not isinstance(val, (int, float)):
                self.logger.error('Unsupported weight initialization value type')
                return
            self.weights = val*self.weights
        elif  method == "random":
            if not isinstance(val, (int, float)):
                self.logger.error('Unsupported weight initialization value type')
                return
            self.weights = val*(1.01-np.random.rand(self.n_base_funcs,self.n_dim))
        elif  method == "external":
            if not (isinstance(val, np.ndarray) and val.shape == (self.n_base_funcs,self.n_dim)):
                self.logger.error('Unsupported weight initialization value type')
                return
            self.weights = copy.copy(val)
        else:
            self.logger.error('Unsupported weight initialization method')
            return

    def init_bias(self, method, val = 1):
        """
        Initialize the first layer bias.
        There are several initialization ways supported:
            lineal      -> Lineal distribution across dimensions
            random      -> Random values within a defined range
            external    -> Specify values

        :param method: Initialization method
        :param val: lineal      -> Unused
                    random      -> Random initialization range
                    external    -> Bias matrix

        """

        self.bias = np.ones((self.n_base_funcs,self.n_dim))
        if method == "lineal":
            if not isinstance(val, (int, float)):
                self.logger.error('Unsupported bias initialization value type')
                return

            n_points_per_dimension = int(np.power(self.n_base_funcs,1/self.n_dim))
            n_base_lin = np.power(n_points_per_dimension,self.n_dim)
            n_base_rand = self.n_base_funcs - n_base_lin

            bias_step = 2/n_points_per_dimension
            bias = -1*np.ones(self.n_dim)+bias_step/2
            dim_cnt = np.zeros(self.n_dim)
            self.bias[0,:] = copy.copy(bias)
            for i in range(1,n_base_lin):
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
                self.bias[i,:] = copy.copy(bias)
            self.bias[n_base_lin:(n_base_lin+n_base_rand),:] = 1-2*np.random.rand(n_base_rand,self.n_dim)
        elif  method == "random":
            if not isinstance(val, (int, float)):
                self.logger.error('Unsupported bias initialization value type')
                return
            self.bias = val*(1-2*np.random.rand(self.n_base_funcs,self.n_dim))
        elif  method == "external":
            if not (isinstance(val, np.ndarray) and val.shape == (self.n_base_funcs,self.n_dim)):
                self.logger.error('Unsupported bias initialization value type')
                return
            self.bias = copy.copy(val)
        else:
            self.logger.error('Unsupported bias initialization method')
            return

    def evaluate_ortogonal_base(self):
        """
        Evaluate the Gram-Schmidt procedure.

        """
        all_okay = self._evaluate_b2b()
        if not all_okay:
            return all_okay
        all_okay &= self._evaluate_o2b()
        return all_okay


    def evaluate_proyection(self, inputs, area, labels):
        """
        Evaluate the data proyection.

        :param inputs: Data input matrix
        :param area:  Data input influence area matrix
        :param labels:  Data labels vector

        """
        self.sigma = np.zeros(self.n_base_funcs)
        if not (isinstance(labels, np.ndarray)):
            self.logger.error('Unsupported labels type')
            return copy.copy(self.sigma)
        n_data = len(labels)
        if not (isinstance(inputs, np.ndarray) and inputs.shape == (n_data,self.n_dim)):
            self.logger.error('Unsupported inputs type')
            return copy.copy(self.sigma)
        if not (isinstance(area, np.ndarray) and area.shape == (n_data,self.n_dim)):
            self.logger.error('Unsupported inputs type')
            return copy.copy(self.sigma)

        # Proyections
        proyections_f2b = np.zeros(self.n_base_funcs)
        proyections_f2o = np.zeros(self.n_base_funcs)
        for i in range(self.n_base_funcs):
            for d in range(n_data):
                data_sum = 1
                for dim in range(self.n_dim):

                    hard_limits = np.zeros(2)
                    hard_limits[0] = np.max([inputs[d][dim] - area[d][dim], -1])
                    hard_limits[1] = np.min([inputs[d][dim] + area[d][dim], 1])

                    b1 = self.bias[i][dim]
                    w1 = self.weights[i][dim]
                    (section_cnt,sections_signs,sections_limits) = _evaluate_integration_sections(soft_limits = [b1], hard_limits = hard_limits)

                    dim_coef = 0
                    for s in range(section_cnt):
                        low_lim = sections_limits[s]
                        up_lim = sections_limits[s+1]

                        w1s = sections_signs[0,s]*w1
                        dim_coef += 1/w1s*np.log(np.abs(1+w1s*(up_lim - b1))) - 1/w1s*np.log(np.abs(1+w1s*(low_lim - b1)))
                    data_sum *=dim_coef
                proyections_f2b[i] += labels[d]*data_sum

        if(np.isnan(proyections_f2b).any()):
            self.logger.warning("F2B PROYECTIONS: NAN ERROR")

        for i in range(0, self.n_base_funcs):
            for j in range(i, -1, -1):
                proyections_f2o[i] += self.betas[i,j]*proyections_f2b[j]

        if(np.isnan(proyections_f2o).any()):
            self.logger.warning("F2O PROYECTIONS: NAN ERROR")

        # Sigmas
        for i in range(self.n_base_funcs):
            for j in range(self.n_base_funcs):
                self.sigma[i] += proyections_f2o[j]*self.betas[j,i]

        if(np.isnan(self.sigma).any()):
            self.logger.warning("SIGMAS: NAN ERROR")

        return copy.copy(self.sigma)


    def run(self, inputs):
        """
        Run the network over the input data.

        :param inputs: Data input matrix

        """
        network_output = np.zeros(1)
        if not (isinstance(inputs, np.ndarray) and inputs.shape == (inputs.shape[0],self.n_dim)):
            self.logger.error('Unsupported inputs type')
            return network_output

        n_data = inputs.shape[0]
        network_output = np.zeros(n_data)
        for d in range(n_data):
            first_layer_output = np.zeros(self.n_base_funcs)
            for n in range(self.n_base_funcs):
                first_layer_output[n] = np.prod(1/(1+np.abs(self.weights[n,:]*(inputs[d,:] - self.bias[n,:]))))

            network_output[d] = np.sum(first_layer_output*self.sigma)
            if network_output[d] > 5:
                network_output[d] = 5
            if network_output[d] < -5:
                network_output[d] = -5


        return network_output

    def _evaluate_b2b(self):
        all_okay = True

        self.proyections_b2b = np.ones((self.n_base_funcs,self.n_base_funcs))
        self.norms_b = np.zeros(self.n_base_funcs)
        for i in range(self.n_base_funcs):
            for j in range(i, self.n_base_funcs):
                for d in range(self.n_dim):
                    b1 = self.bias[i,d]
                    b2 = self.bias[j,d]
                    w1 = self.weights[i,d]
                    w2 = self.weights[j,d]
                    dim_coef = 0

                    if b1==b2 and w1==w2:
                        (section_cnt,sections_signs,sections_limits) = _evaluate_integration_sections(soft_limits = [b1], hard_limits = [-1,1])
                        for s in range(section_cnt):
                            low_lim = sections_limits[s]
                            up_lim = sections_limits[s+1]
                            w1s = sections_signs[0,s]*w1
                            dim_coef += -1/(w1s*(1+w1s*(up_lim - b1))) + 1/(w1s*(1+w1s*(low_lim - b1)))
                    else:
                        (section_cnt,sections_signs,sections_limits) = _evaluate_integration_sections(soft_limits = [b1,b2], hard_limits = [-1,1])

                        for s in range(section_cnt):
                            low_lim = sections_limits[s]
                            up_lim = sections_limits[s+1]

                            w1s = sections_signs[0,s]*w1
                            w2s = sections_signs[1,s]*w2

                            den = -1 + w1s/w2s - w1s*b2 + w1s*b1
                            while den == 0:
                                msg = "Zero in Proyection of i: "+str(i)+" over j: "+str(j)
                                # msg += "\n W1s = "+str(w1s)
                                # msg += "\n W2s = "+str(w2s)
                                # msg += "\n b1 = "+str(b1)
                                # msg += "\n b2 = "+str(b2)
                                self.logger.warning(msg)
                                w2s += 1e-5
                                w1s -= 1e-5
                                den = -1 + w1s/w2s - w1s*b2 + w1s*b1
                            num2 = 1/den
                            num1 = w1s/w2s*num2

                            dim_coef += num1*(1/w1s*np.log(np.abs(1+w1s*(up_lim - b1))) - 1/w1s*np.log(np.abs(1+w1s*(low_lim - b1))))
                            dim_coef -= num2*(1/w2s*np.log(np.abs(1+w2s*(up_lim - b2))) - 1/w2s*np.log(np.abs(1+w2s*(low_lim - b2))))

                    self.proyections_b2b[i,j] *=dim_coef
                self.proyections_b2b[j,i] = self.proyections_b2b[i,j]
            self.norms_b[i] = np.sqrt(self.proyections_b2b[i,i])

        if(np.isnan(self.proyections_b2b).any()):
            self.logger.warning("B2B PROYECTIONS: NAN ERROR")
            all_okay = False

        return all_okay

    def _evaluate_o2b(self):
        all_okay = True

        # Betas
        self.betas = np.zeros((self.n_base_funcs,self.n_base_funcs))
        self.proyections_o2b = np.zeros((self.n_base_funcs,self.n_base_funcs))
        self.norms_o = np.zeros(self.n_base_funcs)

        self.norms_o[0] = self.norms_b[0]
        self.betas[0,0] = 1/self.norms_b[0]
        for i in range(1, self.n_base_funcs):
            self.norms_o[i] = np.pow(self.norms_b[i],2)
            for j in range(i):
                self.proyections_o2b[i,j] = self.proyections_b2b[i,j]
                for k in range(j):
                    self.proyections_o2b[i,j] -= self.proyections_o2b[j,k]*self.proyections_o2b[i,k]
                self.proyections_o2b[i,j] /= self.norms_o[j]
                self.norms_o[i] -= np.pow(self.proyections_o2b[i,j],2)
            self.norms_o[i] = np.sqrt(self.norms_o[i])

            self.betas[i,i] = 1/self.norms_o[i]
            for j in range(i):
                for k in range(j, i):
                    self.betas[i,j] -= self.proyections_o2b[i,k]*self.betas[k,j]
                self.betas[i,j] /= self.norms_o[i]

        if(np.isnan(self.proyections_o2b).any()):
            self.logger.warning("O2B PROYECTIONS: NAN ERROR")
            self.logger.debug(self.proyections_o2b)
            all_okay = False

        if(np.isnan(self.betas).any()):
            self.logger.warning("BETAS: NAN ERROR")
            all_okay = False

        return all_okay



def _evaluate_integration_sections(soft_limits, hard_limits):

    n_soft_limits = len(soft_limits)
    max_sections = n_soft_limits+1
    sections_signs = np.zeros((n_soft_limits,max_sections))
    sections_limits = hard_limits[1]*np.ones(max_sections+1)

    sort_lims = np.sort(soft_limits)
    sort_ind_lims = np.argsort(soft_limits)
    section_cnt = 1
    sections_limits[0] = hard_limits[0]
    for i in range(n_soft_limits):
        if sort_lims[i] <= hard_limits[0]:
            sections_signs[sort_ind_lims[i],:] = np.ones(max_sections)
            continue
        if sort_lims[i] >= hard_limits[1]:
            sections_signs[sort_ind_lims[i],:] = -np.ones(max_sections)
            continue
        sections_signs[sort_ind_lims[i],0:section_cnt] = -np.ones(section_cnt)
        sections_signs[sort_ind_lims[i],section_cnt:max_sections] = np.ones(max_sections-section_cnt)
        sections_limits[section_cnt] = sort_lims[i]
        section_cnt += 1

    return section_cnt, sections_signs, sections_limits

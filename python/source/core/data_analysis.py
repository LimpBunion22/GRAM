import numpy as np
from sklearn.cluster import KMeans

W_COEF = 0.2
MIN_W = 1e-16

def distribute_bases(inputs, n_base_funcs, w_coef = W_COEF):
        """
        Analyse data coordinates to evaluate best bias and weights.

        :param inputs: Data matrix
        :param n_base_funcs: Number of base functions

        """

        weights = np.ones((n_base_funcs,inputs.shape[1]))
        m, d = inputs.shape[0], inputs.shape[1]  # Por ejemplo, 1000 datos y 50 dimensiones
        kmeans = KMeans(n_clusters=n_base_funcs, random_state=0).fit(inputs)

        bias = kmeans.cluster_centers_
        etiquetas = kmeans.labels_
        distancias = np.linalg.norm(inputs - bias[etiquetas], axis=1)
        distancia_media_i = np.zeros(n_base_funcs)
        for i in range(n_base_funcs):
            # Seleccionar las distancias de los puntos que pertenecen al cluster i
            distancias_cluster_i = distancias[etiquetas == i]
            # Calcular la distancia media para el cluster i
            distancia_media_i[i] = np.mean(distancias_cluster_i)
            distancia_media_i[i] = np.max([distancia_media_i[i],MIN_W])
            weights[i,:] *= (1-w_coef)/(distancia_media_i[i]*w_coef)

        return bias,weights


def analyze_data(inputs, n_base_funcs, w_coef = W_COEF):
        """
        Analyse data coordinates to evaluate best bias and weights.

        :param inputs: Data matrix
        :param n_base_funcs: Number of base functions

        """

        weights = np.ones((n_base_funcs,inputs.shape[1]))
        areas = np.zeros(inputs.shape)
        m, d = inputs.shape[0], inputs.shape[1]  # Por ejemplo, 1000 datos y 50 dimensiones
        kmeans = KMeans(n_clusters=n_base_funcs, random_state=0).fit(inputs)

        bias = kmeans.cluster_centers_
        etiquetas = kmeans.labels_
        distancias = np.linalg.norm(inputs - bias[etiquetas], axis=1)
        distancia_media_i = np.zeros(n_base_funcs)
        for i in range(n_base_funcs):
            # Seleccionar las distancias de los puntos que pertenecen al cluster i
            index = np.where(etiquetas == i)[0]
            distancias_cluster_i = distancias[index]
            # Calcular la distancia media para el cluster i
            distancia_media_i[i] = np.mean(distancias_cluster_i)
            distancia_media_i[i] = np.max([distancia_media_i[i],MIN_W])
            weights[i,:] *= (1-w_coef)/(distancia_media_i[i]*w_coef)

            datos_cluster = inputs[index]
            n_data_cluster = len(datos_cluster)
            if n_data_cluster < 2:
                dists = np.zeros(n_base_funcs)
                dists[i] = 1e10
                a = index[0]
                for j in range(n_base_funcs):
                    if j == i:
                        continue
                    dists = np.sqrt(np.sum(np.pow(inputs[a] - bias[j],2))) - distancia_media_i[j]
                areas[a] = np.sqrt(np.min(dists))/2
            else:
                dists = np.zeros((n_data_cluster,n_data_cluster))
                for j in range(len(datos_cluster)-1):
                    dists[j,j] = 1e10
                    a = index[j]
                    for k in range(j+1,len(datos_cluster)):
                        b = index[k]
                        dists[j,k] = np.sum(np.pow(inputs[a] - inputs[b],2))
                        dists[k,j] = dists[j,k]
                    areas[a] = np.sqrt(np.min(dists[j,:]))/2
        return bias,weights,areas


def evaluate_influence_areas(inputs):
        """
        Analyse data coordinates to evaluate influence area.

        :param inputs: Data matrix

        """

        areas = np.zeros(inputs.shape)

        for d in range(inputs.shape[1]):

            index = np.argsort(inputs[:,d])

            a = index[0]
            b = index[1]
            areas[b,d] = np.abs((inputs[b,d] - inputs[a,d])/2)
            areas[a,d] = areas[b,d]
            for i in range(1,inputs.shape[0]):
                a = index[i-1]
                b = index[i]
                areas[b,d] = np.abs((inputs[b,d] - inputs[a,d])/2)
                areas[a,d] = np.min([areas[a,d],areas[b,d]])

        return areas

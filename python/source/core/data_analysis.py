import numpy as np
from sklearn.cluster import KMeans



def distribute_bases(inputs, n_base_funcs):
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
        for i in range(n_base_funcs):
            # Seleccionar las distancias de los puntos que pertenecen al cluster i
            distancias_cluster_i = distancias[etiquetas == i]
            # Calcular la distancia media para el cluster i
            distancia_media_i = np.mean(distancias_cluster_i)
            weights[i,:] *= distancia_media_i

        return bias,weights


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

import numpy as np
from sklearn.cluster import KMeans

W_COEF = 0.2
MIN_W = 1e-5

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
            weights[i,:] *= np.min([(1-w_coef)/(distancia_media_i[i]*w_coef),1e3])

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

            datos_cluster = inputs[index]
            n_data_cluster = len(datos_cluster)
            if n_data_cluster < 2:
                bias [i,:] += 1e-8
                dists = np.zeros(n_base_funcs)
                a = index[0]
                for j in range(n_base_funcs):
                    if j == i:
                        continue
                    dists[j] = np.sqrt(np.sum(np.pow(inputs[a] - bias[j],2))) #- distancia_media_i[j]
                dists[i] = 1e20
                areas[a] = np.sqrt(np.min(dists))/2

                distancia_media_i[i] = np.max([areas[a,0],MIN_W])
                weights[i,:] *= np.min([(1-w_coef)/(distancia_media_i[i]*w_coef),500])/20
            else:
                distancia_media_i[i] = np.mean(distancias_cluster_i)
                distancia_media_i[i] = np.max([distancia_media_i[i],MIN_W])
                weights[i,:] *= np.min([(1-w_coef)/(distancia_media_i[i]*w_coef),500])

                dists = np.zeros((n_data_cluster,n_data_cluster))
                for j in range(len(datos_cluster)-1):
                    dists[j,j] = 1e20
                    a = index[j]
                    for k in range(j+1,len(datos_cluster)):
                        b = index[k]
                        dists[j,k] = np.sum(np.pow(inputs[a] - inputs[b],2))
                        dists[k,j] = dists[j,k]
                    areas[a] = np.sqrt(np.min(dists[j,:]))/2
        return bias,weights,areas


def analyze_data2(inputs, n_base_funcs, w_coef = W_COEF, n_anchors = 10):
        """
        Analyse data coordinates to evaluate best bias and weights.

        :param inputs: Data matrix
        :param n_base_funcs: Number of base functions

        """

        n_data, n_dim = inputs.shape[0], inputs.shape[1]  # Por ejemplo, 1000 datos y 50 dimensiones

        weights = np.ones((n_base_funcs,n_dim))
        bias = np.zeros((n_base_funcs,n_dim))
        areas = np.zeros(inputs.shape)
        base_cnt = 0
        base_per_data = n_base_funcs/n_data

        kmeans = KMeans(n_clusters=n_anchors, random_state=0).fit(inputs)

        anchor_pos = kmeans.cluster_centers_
        etiquetas = kmeans.labels_
        distancias = np.linalg.norm(inputs - anchor_pos[etiquetas], axis=1)
        distancia_media_i = np.zeros(n_anchors)
        for i in range(n_anchors):
            # Seleccionar las distancias de los puntos que pertenecen al cluster i
            index = np.where(etiquetas == i)[0]
            distancias_cluster_i = distancias[index]
            # Calcular la distancia media para el cluster i

            datos_cluster = inputs[index]
            n_data_cluster = len(datos_cluster)
            if n_data_cluster == 1:
                anchor_pos[i,:] += 1e-5*(0.5 - np.random.rand(n_dim))
                dists = np.zeros(n_anchors)
                a = index[0]
                for j in range(n_anchors):
                    if j == i:
                        continue
                    dists[j] = np.sqrt(np.sum(np.pow(inputs[a] - anchor_pos[j],2))) #- distancia_media_i[j]
                dists[i] = 1e20
                areas[a] = np.sqrt(np.min(dists))/2

                distancia_media_i[i] = areas[a,0]
                # distancia_media_i[i] = np.max([areas[a,0],MIN_W])

                if base_per_data>0.5 and base_cnt<n_base_funcs:
                    bias[base_cnt,:] = anchor_pos[i,:]
                    weights[base_cnt,:] *= (1-w_coef)/(distancia_media_i[i]*w_coef)
                    base_cnt +=1
            else:
                distancia_media_i[i] = np.mean(distancias_cluster_i)
                # distancia_media_i[i] = np.max([distancia_media_i[i],MIN_W])

                if base_cnt<n_base_funcs:
                    w_factor =(1-w_coef)/(distancia_media_i[i]*w_coef)
                    n_base_cluster = np.min([int(n_data_cluster*base_per_data),n_base_funcs-base_cnt])
                    for nbc in range(base_cnt,base_cnt+n_base_cluster):
                        bias[nbc,:] = anchor_pos[i,:] + 1/5*distancia_media_i[i]*(0.5 - np.random.rand(n_dim))
                        weights[nbc,:] *= 10*w_factor*(0.01 + np.random.rand(n_dim))*(0.01 + nbc-base_cnt)
                    base_cnt += n_base_cluster

                dists = np.zeros((n_data_cluster,n_data_cluster))
                for j in range(len(datos_cluster)-1):
                    dists[j,j] = 1e10
                    a = index[j]
                    for k in range(j+1,len(datos_cluster)):
                        b = index[k]
                        dists[j,k] = np.sum(np.pow(inputs[a] - inputs[b],2))
                        dists[k,j] = dists[j,k]
                    areas[a] = np.sqrt(np.min(dists[j,:]))/2

        if base_cnt<n_base_funcs:
            w_factor = np.min([(1-w_coef)/(distancia_media_i[i]*w_coef),500])
            n_base_cluster = n_base_funcs-base_cnt
            for nbc in range(base_cnt,base_cnt+n_base_cluster):
                bias[nbc,:] = 1 - 2*np.random.rand(n_dim)
                weights[nbc,:] *= 100*(0.1 + np.random.rand(n_dim))

        return bias,weights,areas

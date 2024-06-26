import matplotlib.pyplot as plt
import numpy as np

# Función para dibujar n vectores
def dibujar_vectores(base,vectores,title,block):

    # Crear una figura y un eje
    fig, ax = plt.subplots()

    # Lista de colores
    colores = plt.cm.viridis(np.linspace(0, 1, len(vectores)))

    # Dibujar cada vector
    for i, vector in enumerate(vectores):
        ax.plot(base, vector, color=colores[i], label=f'Vector {i+1}')

    # Configurar límites del gráfico
    ax.set_xlim(-1, 1)

    # Añadir leyenda
    ax.legend()
    plt.title(title)

    # Mostrar gráfico
    plt.show(block=block)

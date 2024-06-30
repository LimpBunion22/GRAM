import logging
import colorlog
import os

def create_logger():
    # Configuración del logger
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)

    # Formateador para el archivo de logs
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d]'
    )

    # Formateador para la consola con colores
    console_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d]',
        log_colors={
            'DEBUG': 'white',
            'INFO': 'cyan',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
    )

    directorio_common = os.path.dirname(os.path.abspath(__file__))
    directorio_source = os.path.dirname(directorio_common)
    directorio_python = os.path.dirname(directorio_source)
    ruta_directorio = os.path.join(directorio_python, 'logs')
    ruta_logs = os.path.join(ruta_directorio, 'mi_log.log')

    if os.path.exists(ruta_directorio):
        print(f"El directorio {ruta_directorio} existe.")
    else:
        print(f"El directorio {ruta_directorio} no existe.")

    # Manejador para el archivo de logs
    file_handler = logging.FileHandler(ruta_logs)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    # Manejador para la consola
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(console_formatter)

    # Añadir los manejadores al logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

    # Ejemplos de mensajes de log
    # logger.debug('Este es un mensaje de depuración.')
    # logger.info('Este es un mensaje informativo.')
    # logger.warning('Este es un mensaje de advertencia.')
    # logger.error('Este es un mensaje de error.')
    # logger.critical('Este es un mensaje crítico.')

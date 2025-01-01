"""
Script para descargar imágenes de Google para el dataset CIFAR-10
Requiere: pip install icrawler
"""

from icrawler.builtin import GoogleImageCrawler
import os

# Configuración de las categorías y términos de búsqueda
categories = {
    'avion': ['avion comercial', 'avion vista lateral', 'airplane side view'],
    'automovil': ['automovil vista lateral', 'coche vista lateral', 'car side view'],
    'pajaro': ['pajaro entero', 'bird full body', 'bird side view'],
    'gato': ['gato entero', 'cat full body', 'cat side view'],
    'ciervo': ['ciervo entero', 'deer full body', 'deer side view'],
    'perro': ['perro entero', 'dog full body', 'dog side view'],
    'rana': ['rana verde', 'frog full body', 'frog side view'],
    'caballo': ['caballo entero', 'horse full body', 'horse side view'],
    'barco': ['barco vista lateral', 'ship side view', 'boat side view'],
    'camion': ['camion vista lateral', 'truck side view', 'lorry side view']
}

# Número de imágenes a descargar por término de búsqueda
IMAGES_PER_SEARCH = 10

def download_images():
    base_dir = 'dataset_propio'

    # Crear directorio base si no existe
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Iterar sobre cada categoría
    for category, search_terms in categories.items():
        print(f"\nDescargando imágenes para: {category}")
        category_dir = os.path.join(base_dir, category)

        # Crear directorio de categoría si no existe
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)

        # Descargar imágenes para cada término de búsqueda
        for term in search_terms:
            crawler = GoogleImageCrawler(
                storage={'root_dir': category_dir},
                feeder_threads=1,
                parser_threads=1,
                downloader_threads=4
            )

            # Configurar filtros para obtener imágenes de mejor calidad
            filters = dict(
                size='medium',
                type='photo',
                license='commercial,modify'
            )

            print(f"  Buscando: {term}")
            crawler.crawl(
                keyword=term,
                filters=filters,
                max_num=IMAGES_PER_SEARCH,
                file_idx_offset='auto'
            )

if __name__ == "__main__":
    print("Iniciando descarga de imágenes...")
    download_images()
    print("\n¡Descarga completada!")
    print("Nota: Por favor, revisa manualmente las imágenes y selecciona las 15 mejores para cada categoría.")

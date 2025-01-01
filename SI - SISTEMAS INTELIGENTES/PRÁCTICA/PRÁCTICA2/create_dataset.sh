#!/bin/bash

# Crear directorio principal
mkdir -p dataset_propio

# Lista de categorías
categories=(
    "avion"
    "automovil"
    "pajaro"
    "gato"
    "ciervo"
    "perro"
    "rana"
    "caballo"
    "barco"
    "camion"
)

# Crear subdirectorios para cada categoría
for category in "${categories[@]}"; do
    mkdir -p "dataset_propio/$category"
done

echo "Estructura de directorios creada correctamente."

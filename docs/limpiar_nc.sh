#!/bin/bash

# --- CONFIGURACIÓN ---
# Define el directorio principal que contiene las carpetas por fecha (ej. 20061017, 20061018, etc.)
# Asegúrate de que esta ruta sea la correcta.
BASE_DIR="/home/f-caballero/UM/TIF3/MDV_para_15_Octubre/netCDF/2012"

# --- INICIO DEL SCRIPT ---

# 1. Verificar si el directorio base existe para evitar errores.
if [ ! -d "$BASE_DIR" ]; then
  echo "Error: El directorio base '$BASE_DIR' no fue encontrado."
  exit 1
fi

echo "✅ Iniciando el proceso en el directorio: $BASE_DIR"

# 2. Recorrer cada subdirectorio dentro del directorio base.
# El patrón */ se asegura de que solo procesemos las carpetas.
for dir in "$BASE_DIR"/*/; do

  # Comprobamos que realmente sea un directorio.
  if [ -d "$dir" ]; then
    echo ""
    echo "📁 Procesando directorio: $dir"

    # 3. Eliminar todos los archivos que NO terminan en .nc
    # El comando 'find' busca archivos (-type f) cuyo nombre NO coincida con "*.nc"
    # y los elimina de forma segura con la opción -delete.
    find "$dir" -type f ! -name "*.nc" -delete
    echo "    - Archivos basura eliminados."

    # 4. Renombrar los archivos .nc restantes.
    # Recorremos todos los archivos que terminan en .nc dentro del directorio actual.
    for file in "$dir"*.nc; do
    
      # Esta comprobación previene errores si una carpeta no tuviera archivos .nc
      if [ -f "$file" ]; then
        # Obtenemos solo el nombre del archivo, sin la ruta.
        filename=$(basename "$file")
        
        # Usamos una expansión de parámetros de Bash para obtener todo lo que está
        # después del primer guion bajo (_).
        # Ejemplo: de "ncfdata20061017_185933.nc" se obtiene "185933.nc"
        new_filename="${filename#*_}"
        
        # Renombramos el archivo original a su nuevo nombre.
        mv "$file" "$dir$new_filename"
        
      fi
    done
    echo "    - Archivos .nc renombrados correctamente."
  fi
done

echo ""
echo "✨ ¡Proceso completado con exito!"

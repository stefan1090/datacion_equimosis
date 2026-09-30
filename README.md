# Análisis de datación de equimosis

Código del estudio «Evaluación de modelos de Deep Learning para la Datación de Equimosis en el ámbito de la Medicina Legal en el Perú».

Incluye el análisis principal (<6 frente a 6–24 horas), la sensibilidad con corte de 12 horas y el análisis del relleno de las imágenes. Se incluyen participantes de 18 años o más, con evolución conocida de hasta 24 horas. Las tablas y figuras se generan al ejecutar el código; no se duplican aquí los resultados del artículo.

## Archivos

- `preparar_base.py`: convierte el Excel curado del estudio en el CSV analítico, comprobando fechas, tiempos e identificadores.
- `extraer_caracteristicas.py`: representaciones de ResNet18, EfficientNet-B0 y MNASNet 1.0, y características de color/textura.
- `analizar.py`: modelos de imagen sola e imagen + datos, referencias, métricas, curvas ROC y matrices de confusión para ambos cortes.
- `intervalos_6h.py`: IC95% de exactitud balanceada y AUROC media del análisis principal, con 10000 réplicas.
- `analizar_padding.py`: sensibilidad al relleno >50% y asociación entre relleno y categoría temporal, para 6 horas.
- `metodos.py`: funciones compartidas y validación de las entradas.

## Instalación y datos

Usar Python 3.12 e instalar `requirements.txt` para el entrenamiento de clasificadores y la evaluación:

```bash
python -m pip install -r requirements.txt
```

Las entradas se mantienen en una carpeta privada, por ejemplo `datos_privados/estudio/`. Si se parte del Excel curado (hoja `Dataset`), generar el CSV así:

```bash
python preparar_base.py --excel datos_privados/base_equimosis.xlsx --salida datos_privados/estudio
```

La hoja debe contener los campos indicados en el código, incluidas ambas fechas de lesión y captura. Este paso comprueba que el tiempo coincide con las fechas y omite registros sin tiempo. **No realiza la selección clínica de lesiones ni recorta fotografías**: esas decisiones ya deben estar aplicadas al Excel y a la carpeta de imágenes. Después se usan estas entradas:

- `base_analitica.csv`: una fila por imagen, con columnas `persona_id,archivo,tiempo_horas,edad,sexo,localizacion,orientacion`. CSV UTF-8, coma como separador y punto decimal; sin valores faltantes. Edad, sexo y tiempo son constantes dentro de cada persona.
- `imagenes/`: recortes revisados, con los nombres indicados en `archivo`.
- `particiones_6h.json`: registro histórico privado de las cinco particiones. Lista de objetos con `grupo` (1 a 5), `personas_entrenamiento` y `personas_evaluacion` (listas de identificadores). Cada persona se evalúa una sola vez y nunca aparece simultáneamente en entrenamiento y evaluación.
- `caracteristicas/`: caché privada original, o creada con la orden siguiente. Contiene los cuatro archivos `*_caracteristicas.npz` (campos `X`, `archivo`, `persona_id`) y `huellas_imagenes.csv` (campos `archivo`, `sha256`).

Para extraer características desde imágenes, instalar **en un entorno separado** `requirements-extraccion.txt`, que conserva las versiones del procesamiento original. La primera extracción descarga los pesos ImageNet V1:

```bash
python -m pip install -r requirements-extraccion.txt
python extraer_caracteristicas.py --proyecto datos_privados/estudio --salida datos_privados/estudio/caracteristicas
```

## Ejecución

Ejecutar desde esta carpeta, con el entorno de `requirements.txt` activo. Las carpetas de salida deben ser nuevas.

```bash
# Principal de 6 horas, conservando las particiones históricas.
python analizar.py --proyecto datos_privados/estudio --caracteristicas datos_privados/estudio/caracteristicas --corte 6 --particiones datos_privados/estudio/particiones_6h.json --salida salidas_privadas/principal_6h

# Intervalos de confianza del principal; --todos incluye las doce variantes.
python intervalos_6h.py --resultados salidas_privadas/principal_6h --salida salidas_privadas/IC_6h --bootstrap 10000 --todos --sin-mostrar

# Sensibilidad de 12 horas: nueva estratificación por persona y clase, semilla 42.
python analizar.py --proyecto datos_privados/estudio --caracteristicas datos_privados/estudio/caracteristicas --corte 12 --salida salidas_privadas/sensibilidad_12h

# Relleno: EfficientNet-B0 y MNASNet, sobre el principal de 6 horas.
python analizar_padding.py --proyecto datos_privados/estudio --caracteristicas datos_privados/estudio/caracteristicas --principal salidas_privadas/principal_6h --salida salidas_privadas/padding_6h
```

En Colab, estas entradas también pueden ejecutarse con `%run` y los mismos argumentos. `--crear-particiones` permite una nueva ejecución de 6 horas si no se dispone del registro histórico, pero no reproduce la asignación utilizada en el estudio. El resumen de IC de 6 h comprueba la cohorte del informe: 35 participantes y 74 imágenes.

## Configuración estadística

Cinco particiones por participante y el mismo peso total para cada persona. Las redes permanecen congeladas; se ajustan clasificadores logísticos (`C=0.01`) o SVM RBF (C entre 0.1, 1 y 10, seleccionado en tres particiones internas). Los transformadores se ajustan solo en entrenamiento. Se evalúan cinco estrategias de imagen, con y sin edad, sexo, localización y orientación, más las referencias mayoritaria y de datos solos.

El relleno de las redes usa RGB (124, 116, 104) y tamaño 224 × 224; color/textura no utiliza relleno. La AUROC media es la media de las cinco AUROC por partición. Los IC son percentiles de bootstrap por persona sobre predicciones fijas, sin reentrenamiento: 10000 réplicas para BA y AUROC del principal; 2000 para BA de 12 h, aporte de datos y escenarios de relleno; 5000 para asociación relleno/clase. El bootstrap de AUROC media se estratifica por partición y clase; el de BA, por clase. Las sensibilidades son exploratorias y los IC no incorporan toda la incertidumbre del entrenamiento ni de las múltiples comparaciones.

## Disponibilidad y verificación

No se distribuyen fotografías, datos individuales, características extraídas ni modelos ajustados con los participantes. Las salidas contienen información individual y se mantienen privadas. Reproducir los valores del estudio requiere acceso autorizado a los datos y a las particiones históricas; el código por sí solo no los reemplaza.

Esta distribución reprodujo las predicciones de las doce variantes de ambos cortes y las tablas de intervalos y relleno usando las características originales. La extracción de las redes no se volvió a ejecutar en esa comprobación. Código propio bajo licencia MIT; datos y componentes de terceros conservan sus condiciones respectivas.

# Decision Transformer para Sistemas de Recomendación

-  Trabajo realizado en el marco de la materia Aprendizaje por Refuerzos brindada por la Diplomatura en Ciencia de Datos (2025)

## Estructura del repositorio
```text
Decision-Transformer-Recomendaciones/
├── data
│   ├── groups
│   ├── processed
│   ├── test_users
│   └── train
├── models
├── notebooks
│   └── checkpoints
├── reference_code
│   └── checkpoints
├── results
│   ├── trained_models
│   └── training_histories
└── src
    ├── data
    ├── evaluation
    ├── models
    └── training
```


### Instalación

El proyecto utiliza las siguientes librerías :

- Python 3.11+
- PyTorch
- NumPy
- Pandas
- Matplotlib


El entorno completo puede instalarse utilizando el archivo `requirements.yml`.
En caso de ejectuar el entorno en Windows, utilizar `requirements_windows.yml` para garantizar su compatibilidad.


```bash
conda env create -f requirements.yml
```
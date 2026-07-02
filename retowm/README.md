# retoWM

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Requisitos

- Python 3.13
- [uv](https://github.com/astral-sh/uv)

## Instalación

```bash
# 1. Crear el ambiente virtual
uv venv .venv --python 3.13
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 2. Instalar dependencias
uv pip install -r requirements.txt

# 3. Instalar el proyecto como paquete
pip install -e .
```

> `requirements.txt` contiene todas las dependencias con versiones exactas para reproducibilidad.
> `pyproject.toml` es la configuración interna de Kedro y se usa en el paso 3 para registrar el paquete `retowm`.

## Cómo correr el pipeline

```bash
kedro run
```

## Cómo correr los tests

```bash
pytest
```

## Notebooks

Para explorar los datos con acceso al catálogo de Kedro:

```bash
kedro jupyter lab
```

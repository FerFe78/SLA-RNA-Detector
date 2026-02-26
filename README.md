# Detector de Stem-Loop A (SLA)

Este proyecto implementa un sistema para la detección de estructuras Stem-Loop A (SLA) en secuencias de ARN, utilizando representaciones basadas en modelos de lenguaje de proteínas (RNA-FM) y técnicas de visión artificial para la clasificación.

## Estructura del Proyecto

- `src/`: Código fuente del proyecto, incluyendo la aplicación Streamlit, scripts de procesamiento y el paquete `sla_detector`.
- `data/`: Conjuntos de datos utilizados para el entrenamiento y validación.
- `docs/`: Documentación técnica, artículos y presentaciones relacionadas con el proyecto.
- `results/`: Gráficos, curvas de aprendizaje y visualizaciones de los resultados obtenidos.

## Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/tu-usuario/Proyecto_Integrador_IA.git
   cd Proyecto_Integrador_IA
   ```

2. Crea un entorno virtual e instala las dependencias:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Uso

Para ejecutar la interfaz de usuario (Streamlit):
```bash
streamlit run src/app.py
```

## Autores

- [Tu Nombre/Usuario]

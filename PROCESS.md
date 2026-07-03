# PROCESS.md

## Objetivo de este documento

Este archivo documenta el proceso seguido para resolver el reto. Incluye la definición del problema, las decisiones técnicas, el diseño de validación, los trade-offs considerados, las herramientas utilizadas y el uso de IA. Su propósito es hacer transparente y reproducible el razonamiento detrás de la solución.

---

## Comprensión del reto

El reto proporcionó datos operacionales de retail y dejó al candidato la libertad de definir el problema de negocio, el enfoque técnico y el método de validación. La decisión de diseño más temprana fue construir una solución reproducible con pipelines estructurados en lugar de un notebook exploratorio único. Esto permitiría evaluar la solución tanto desde una perspectiva técnica como de negocio.

---

## Entendimiento inicial de los datos

**Fuentes disponibles:**

- `transactions.csv`: ventas diarias al nivel fecha × tienda × categoría.
- `stores.csv`: atributos de cada tienda (formato, región, nivel socioeconómico, etc.).
- `calendar.csv`: variables de calendario, festivos y eventos comerciales.

**Observaciones relevantes:**

- El dataset cubre aproximadamente 14 meses (enero 2023 – febrero 2024).
- Contiene 80 tiendas y 6 categorías.
- Algunos campos presentan valores nulos: `units_sold`, `avg_ticket`, `replenishment_signal`.
- Algunas combinaciones fecha × tienda × categoría están ausentes, aunque la cobertura es casi completa.

**Variables candidatas como objetivo:**

Se consideraron `amount_total`, `units_sold`, `total_transactions`, `avg_ticket` y `replenishment_signal`. Se seleccionó `amount_total` porque está directamente vinculada al pronóstico de ventas, no presenta valores nulos en la tabla de modelado, permite validación cuantitativa contra una línea base y es la variable más directamente orientada al negocio.

---

## Definición del problema de negocio

Se evaluaron varias formulaciones del problema:

- Pronóstico de ventas diarias.
- Priorización comercial / identificación de oportunidades.
- Detección de anomalías.
- Predicción de señal de reposición.

Se seleccionó el **pronóstico de ventas diarias** porque los datos soportan naturalmente esta formulación, permite validación limpia contra datos futuros, y apoya planificación de demanda, monitoreo comercial y priorización por tienda-categoría. Se evitó formular directamente un problema de inventario o reposición porque los datos de inventario, precio, margen y stockouts no están disponibles.

La detección de anomalías y la predicción de señal de reposición son extensiones futuras válidas, pero no la mejor formulación primaria dado el dataset disponible.

---

## Diseño técnico

Se utilizó **Kedro** para estructurar la solución como pipelines reproducibles con una arquitectura medallion simplificada. Esta separación permite ejecutar cada etapa de forma independiente, facilita la inspección de artefactos intermedios y hace explícitas las dependencias entre capas. Los pipelines implementados y su responsabilidad se describen en el `README.md`.

La solución no incluye capas de producción como API serving, orquestación o registro de modelos porque el alcance del reto es un ejercicio analítico. El énfasis está en reproducibilidad, validación limpia y resultados interpretables.

---

## Ingeniería de variables y prevención de leakage

- Los lags se calcularon por grupo `store_id × category` usando `shift`, de modo que el valor del día actual del objetivo no se usa para predecirse a sí mismo.
- Las rolling features utilizan valores shifteados por la misma razón.
- `date` y `date_dt` no se usan como features directas del modelo.
- `amount_total` es el objetivo y no es predictor directo.
- El preprocesamiento categórico (One-Hot Encoding) se ajusta únicamente sobre los datos de entrenamiento a través del pipeline de scikit-learn.

**Variables excluidas por riesgo de leakage:**

`amount_cash`, `amount_card`, `total_transactions`, `cash_transactions`, `card_transactions`, `units_sold`, `avg_ticket`, `replenishment_signal`.

Estas variables son contemporáneas o pueden no estar disponibles al momento de generar predicciones. Los lags y rolling features sí se incluyen porque se basan exclusivamente en valores históricos de `amount_total`.

---

## Estrategia de validación

El split es estrictamente temporal:

- **Entrenamiento:** 2023-01-01 a 2023-12-31.
- **Prueba:** 2024-01-01 a 2024-02-29.

**Por qué no se usó KFold aleatorio:**
Mezclaría observaciones futuras y pasadas, introduciendo leakage temporal.

**Por qué no se usó walk-forward CV:**
La historia disponible es de aproximadamente 14 meses. Eventos como Buen Fin y Navidad aparecen una sola vez en el periodo de entrenamiento. Folds parciales (ene-jun, ene-ago, ene-oct) tendrían distribuciones de eventos incomparables entre sí, y promediar sus métricas produciría una estimación inestable o engañosa. Dividir 2023 adicionalmente dejaría patrones de eventos importantes fuera del entrenamiento.

**El modelo se entrenó una sola vez** sobre el periodo completo de 2023 y se evaluó una sola vez sobre el holdout futuro. No se realizó tuning sobre el conjunto de prueba ni se utilizó un split interno de validación. Esta fue una decisión metodológica deliberada, no una omisión.

---

## Selección de la línea base

La línea base operacional es `amount_total_lag_7`: predice cada combinación tienda-categoría-día usando el valor de la misma tienda-categoría del mismo día de la semana anterior. Es una referencia razonable para retail porque las ventas diarias frecuentemente exhiben estacionalidad semanal. Un modelo solo es útil si mejora sobre esta regla simple.

---

## Selección del modelo

**Por qué LightGBM:**
El problema se formuló como un modelo tabular global entrenado sobre todas las series tienda × categoría simultáneamente. LightGBM es apropiado para datos tabulares con lags, rolling features, variables de calendario y atributos de tienda/categoría. Captura no linealidades e interacciones de forma eficiente y evita la complejidad de ajustar un modelo independiente por cada serie.

**Por qué no Prophet:**
El dataset tiene 480 series tienda × categoría. Prophet requeriría típicamente muchos modelos independientes o una configuración más compleja. La historia disponible es corta para estimar estacionalidad anual de forma robusta. El enfoque tabular global está más alineado con el conjunto de features disponible.

**Por qué no se compararon múltiples modelos:**
El objetivo no era hacer model shopping. No existía un conjunto de validación independiente para selección de modelos sin sacrificar datos de eventos importantes. Comparar muchos modelos en el conjunto de prueba implicaría tuning implícito sobre ese conjunto. La solución priorizó un modelo sólido versus una línea base operacional.

**Hiperparámetros utilizados (fijos, no optimizados):**

```
n_estimators: 300
learning_rate: 0.05
num_leaves: 31
min_child_samples: 50
subsample: 0.8
colsample_bytree: 0.8
random_state: 42
num_threads: 1
```

Los parámetros fueron elegidos para controlar la complejidad del modelo y reducir el riesgo de sobreajuste. No fueron optimizados contra el conjunto de prueba. `num_threads: 1` garantiza resultados deterministas entre ejecuciones.

---

## Resultados e interpretación

LightGBM reduce el MAE aproximadamente un 13% respecto a la línea base lag_7. Las métricas completas, el análisis por segmento y la importancia de variables se encuentran en `notebooks/03_technical_results.ipynb`; los resultados orientados a negocio en `notebooks/02_business_results.ipynb`.

Los resultados representan reducción del error de pronóstico sobre `amount_total`, no impacto directo en utilidad, margen, ahorro en inventario ni ventas recuperadas. Cuantificar ese impacto requeriría datos de inventario, precio, margen, stockouts y costos operacionales que no están disponibles. La importancia de variables refleja relevancia predictiva, no efecto causal.

---

## Presentación de resultados

Se crearon dos notebooks de resultados:

- **`02_business_results.ipynb`**: orientado a ejecutivos y stakeholders de negocio. Presenta la reducción del error de pronóstico, áreas prioritarias y limitaciones en lenguaje no técnico.
- **`03_technical_results.ipynb`**: orientado a Data Scientists e Ingenieros de ML. Cubre métricas detalladas, estrategia de validación, prevención de leakage, importancia de variables y análisis sistemático de errores.

Esta separación evita sobrecargar a cada audiencia con los detalles relevantes para la otra.

---

## Limitaciones

- Solo están disponibles aproximadamente 14 meses de historia.
- El periodo de prueba cubre únicamente enero–febrero 2024.
- Buen Fin y Semana Santa no tienen observaciones positivas en el periodo de prueba.
- No se dispone de datos de inventario, precio, margen, stockouts, merma, shrink ni costos operacionales.
- La reducción del error de pronóstico no es impacto financiero directo.
- No se realizó CV ni tuning por las restricciones de historia y la unicidad de los eventos anuales.
- El modelo debe revalidarse con más historia antes de un despliegue operacional.
- La importancia de variables no implica causalidad.

---

## Trabajo futuro

- Validar con horizonte más largo y más ciclos comerciales completos.
- Comparar LightGBM, XGBoost y CatBoost bajo un diseño de validación temporal apropiado.
- Incorporar variables de precio, margen, inventario, stockouts, campaña y factores externos.
- Evaluar si las mejoras en pronóstico se traducen en beneficios en inventario, reposición, margen o ventas perdidas.
- Agregar seguimiento formal de experimentos con MLflow si se realizan múltiples experimentos, tuning o versionado de modelos.
- Integrar versionado de datos con DVC cuando el dataset crezca con nuevas transacciones periódicas.
- Migrar la capa Gold a Delta Lake en un escenario de producción con actualizaciones incrementales diarias, para habilitar time travel y garantías ACID.
- Incorporar orquestación o despliegue solo si el caso de uso avanza hacia producción.

---

## Uso de herramientas e IA

**Stack técnico utilizado:** Python, pandas, scikit-learn, LightGBM y Kedro.

**Uso de asistencia de IA:** durante el desarrollo se utilizó asistencia de IA (Claude) como herramienta de productividad. El candidato definió el enfoque del problema, diseñó la estrategia de validación, seleccionó la arquitectura de pipelines y tomó todas las decisiones metodológicas. La asistencia de IA se empleó para acelerar la implementación, revisar código, refinar documentación y cuestionar activamente decisiones de diseño — incluyendo el uso de CV, Prophet, MLflow y modelos adicionales. Todos los outputs fueron validados por el candidato contra el problema de negocio y las restricciones metodológicas del proyecto.

---

## Reproducibilidad

Las instrucciones de instalación y los comandos de ejecución se encuentran en el `README.md`. Los notebooks leen artefactos generados desde `data/07_model_output/` y `data/08_reporting/`; es necesario ejecutar el pipeline de Kedro antes de abrirlos.

---

## Nota final

La solución es una línea base reproducible para pronóstico de ventas retail. Prioriza validación temporal limpia, prevención de leakage, interpretabilidad y relevancia de negocio sobre tuning agresivo del modelo. El proyecto está diseñado para extenderse a medida que se disponga de datos operacionales más ricos.

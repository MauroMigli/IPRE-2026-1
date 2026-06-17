# Pipeline de Análisis Estadístico: dDTF + TFCE 4D con Permutaciones

Este documento describe el pipeline completo para procesar matrices de conectividad funcional calculadas mediante **dDTF (directed Directed Transfer Function)** utilizando **TFCE (Threshold-Free Cluster Enhancement)** adaptado a un espacio tetradimensional, controlando la tasa de falsos positivos (FWER) mediante pruebas de permutación de estadístico máximo (Monte Carlo empírico).

---

## 1. Espacio de Estados y Estructura de Datos

Se asume que las matrices dDTF ya están calculadas. Para cada sujeto $s$ y cada condición experimental, los datos se estructuran en un espacio topológico discreto de 4 dimensiones $\mathcal{P} = (i, j, f, t)$:
- $i$: Nodo Destino (Canal EEG de llegada)
- $j$: Nodo Fuente (Canal EEG de origen)
- $f$: Frecuencia (p. ej., de 1 a 45 Hz en pasos de 1 Hz)
- $t$: Época / Ventana de tiempo (p. ej., ventanas deslizantes)

### Diseño Experimental (Mixto 2x2)
- **Factor Intra-Sujeto:** Estímulo (Nivel 1: `heartbeat`, Nivel 2: `nothing`)
- **Factor Inter-Sujeto:** Condición del Lactante (Grupo 1: `Full-term`, Grupo 2: `Pre-term`)

---

## 2. Fase 1: Mapa del Estadístico de Prueba Base $T(v)$

El objetivo es evaluar la **interacción** del diseño mixto (si la respuesta de red al latido cardíaco difiere según la edad gestacional).

### Paso 1.1: Contraste de Primer Nivel (Intra-Sujeto)
Para cada sujeto $s$ de manera independiente, y para cada coordenada $v = (i,j,f,t)$, se calcula la diferencia del estímulo. 
*Nota: Dado que dDTF $\in [0,1]$, se recomienda aplicar opcionalmente una transformación estabilizadora de la varianza (Fisher/Arcsine) antes de restar.*

$$\Delta_s(v) = dDTF_{s, \text{heartbeat}}(v) - dDTF_{s, \text{nothing}}(v)$$

### Paso 1.2: Contraste de Segundo Nivel (Inter-Sujeto)
Se dividen los vectores de diferencias en los dos grupos independientes (`Full-term` con tamaño $n_{Full}$ y `Pre-term` con tamaño $n_{Pre}$). Se calcula un **Test t de Welch** (asumiendo varianzas desiguales) en cada coordenada $v$ de forma vectorizada usando `scipy.stats.ttest_ind(..., equal_var=False)`:

$$T(v) = \frac{\bar{\Delta}_{Full}(v) - \bar{\Delta}_{Pre}(v)}{\sqrt{\frac{s_{Full}^2(v)}{n_{Full}} + \frac{s_{Pre}^2(v)}{n_{Pre}}}}$$

Donde $\bar{\Delta}$ es la media muestral y $s^2$ es la varianza muestral del grupo en esa coordenada. El resultado es un mapa base de estadísticos $T_0 \in \mathbb{R}^{|\mathcal{P}|}$.

---

## 3. Fase 2: Transformación TFCE (Threshold-Free Cluster Enhancement)

En lugar de binarizar el mapa $T_0$ con un umbral estático, calculamos el soporte topológico continuo para cada punto mediante la representación *Layer-Cake* (Teorema de Fubini-Tonelli), integrando sobre el eje del umbral $h$.

### Paso 2.1: Definición de Adjacencia de Red (Grafo de Vecindad)
Dos coordenadas en el espacio 4D $v = (i,j,f,t)$ y $v' = (i',j',f',t')$ son vecinas si cumplen simultáneamente continuidad temporal, espectral y cercanía espacial de aristas:
1. **Temporal:** $|t - t'| \le 1$
2. **Espectral:** $|f - f'| \le 1$
3. **Espacial (Bola Métrica $B_R$):** Los terminales de los vectores de conectividad deben caer dentro de un radio físico $R$ en el cuero cabelludo:
   $$i' \in B_R(i) \quad \text{AND} \quad j' \in B_R(j)$$

### Paso 2.2: Búsqueda de Componentes Conexas y Extensión $e(h, p)$
Para un umbral $h$ determinado, se define el subgrafo suprathreshold:
$$V_h = \{v \in \mathcal{P} \mid T(v) \ge h\}$$
La extensión $e(h, p)$ es la **cardinalidad** (número de vértices) de la componente conexa máxima dentro de $V_h$ que contiene al punto $p$. Si $T(p) < h$, entonces $e(h, p) = 0$.

### Paso 2.3: Cálculo Numérico de la Integral TFCE
La fórmula matemática continua es:
$$\text{TFCE}(p) = \int_{0}^{T(p)} e(h, p)^E \cdot h^H \, dh$$

En código, esto se aproxima mediante una discretización del umbral $h$ en pasos finitos $\Delta h$ (p. ej., $\Delta h = 0.1$) desde $0$ hasta $\max(T_0)$, implementado como una suma de Riemann:

$$\text{TFCE}(p) \approx \sum_{k=1}^{M} e(k\Delta h, p)^E \cdot (k\Delta h)^H \cdot \Delta h$$

*Hiperparámetros estándar:* $E = 0.5$ (ponderación de extensión de red) y $H = 2.0$ (ponderación de altura/intensidad focal). 
Almacenar este mapa como $TFCE_{real} \in \mathbb{R}^{|\mathcal{P}|}$.

---

## 4. Fase 3: Pruebas de Permutación Monte Carlo y Control FWER

Para obtener p-valores corregidos por comparaciones múltiples que respeten la autocorrelación de la señal, se utiliza el método del Estadístico Máximo (Supremo del campo aleatorio).

### Algoritmo de Identificación de Significancia

1. **Inicializar** una lista vacía para almacenar los máximos bajo la hipótesis nula: $\mathcal{M} = [ ]$.
2. **Definir el número de permutaciones** $K$ (mínimo recomendado $K = 1000$).
3. **Bucle de Permutaciones:** Para $k = 1$ hasta $K$:
   - **Mezclar etiquetas:** Aleatorizar las asignaciones de los sujetos a los grupos `Full-term` y `Pre-term` (rompiendo el efecto real pero manteniendo intactas las matrices de conectividad de cada sujeto).
   - **Calcular mapa T permutado:** Ejecutar el Test t de Welch (Paso 1.2) con las etiquetas barajadas para obtener $T^{(k)}(v)$.
   - **Calcular mapa TFCE permutado:** Ejecutar la suma de Riemann (Paso 2.3) sobre el mapa $T^{(k)}$ para obtener $TFCE^{(k)}(v)$.
   - **Extraer el Supremo:** Encontrar el valor máximo absoluto en todo el mapa 4D de esta iteración:
     $$M^{(k)} = \max_{v \in \mathcal{P}} TFCE^{(k)}(v)$$
   - **Guardar:** Añadir $M^{(k)}$ a la lista $\mathcal{M}$.
4. **Calcular P-valores Corregidos ($p_{corr}$):** Para cada una de las coordenadas originales del espacio 4D, evaluar cuántos supremos de la distribución nula lograron superar o igualar el valor medido en el experimento real:
   $$p_{corr}(p) = \frac{1 + \sum_{k=1}^{K} \mathbb{1}[M^{(k)} \ge TFCE_{real}(p)]}{1 + K}$$

---

## 5. Instrucciones para la Implementación en Código

El agente autónomo debe estructurar el programa utilizando las siguientes librerías y estrategias de optimización:

1. **Estructuras de datos primarias:** Utilizar arreglos multidimensionales de `numpy` de forma integrada (`shape = (n_sujetos, n_destinos, n_fuentes, n_frecuencias, n_epocas)`).
2. **Cálculo de Tests T masivos:** Evitar bucles `for` nativos de Python para los tests estadísticos. Utilizar el parámetro `axis` en `scipy.stats.ttest_ind(..., axis=0, equal_var=False)` para procesar las miles de aristas en paralelo.
3. **Optimización del Flood-Fill (Componentes Conexas):** Para el cálculo de $e(h,p)$, la vecindad espacial está definida por la matriz de distancias geodésicas/euclidianas de los canales EEG. Se sugiere precalcular una matriz de adyacencia booleana basada en el umbral del radio $R$. Utilizar `scipy.ndimage.label` o implementaciones personalizadas de grafos (`networkx` o matrices de dispersión en `scipy.cluster`) para extraer los tamaños de los componentes a cada nivel de $h$ de manera eficiente.
4. **Paralelización de Permutaciones:** Dado que el bucle de permutaciones $K$ es un problema "vergonzosamente paralelo" (*embarrassingly parallel*), se debe implementar usando `joblib` o `multiprocessing` para distribuir las iteraciones entre los núcleos de CPU disponibles en el sistema.

---
**Resultado esperado:** El script final debe exportar un mapa 4D de dimensiones `(n_destinos, n_fuentes, n_frecuencias, n_epocas)` que contenga los valores de $p_{corr}$. Los puntos con $p_{corr} < 0.05$ se consideran estadísticamente significativos bajo control estricto de FWER.

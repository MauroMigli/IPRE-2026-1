# Contexto y Objetivos del Proyecto: TFCE + dDTF 4D

Estoy realizando una investigación de pregrado enfocada en el desarrollo y validación de métodos estadísticos para cuantificar la conectividad neuronal. El pipeline utiliza matrices de conectividad funcional calculadas mediante **dDTF (directed Directed Transfer Function)** sobre un espacio tetradimensional `(source, destination, frequency, epoch)`. Para evaluar la significancia estadística controlando estrictamente la tasa de falsos positivos globales (**FWER - Family-Wise Error Rate**), empleamos el método de **TFCE (Threshold-Free Cluster Enhancement)** combinado con pruebas de permutación no paramétricas (Monte Carlo empírico entre sujetos).

Actualmente necesito cumplir dos grandes objetivos para adaptar el pipeline a los nuevos requerimientos y evaluar la sensibilidad del algoritmo de clústeres.

---

## 🎯 Tareas a Realizar

### Tarea 1: Refactorización del Flujo de Entrada (`main.py`)
- **Archivos de Referencia:** Por favor, lee detalladamente los archivos locales `instructions.md` (donde se encuentra documentada la base técnica actual del flujo dDTF) y `context.md` (donde se especifican las nuevas instrucciones organizacionales y rutas de datos entregadas por la profesora).
- **Acción:** Reemplaza o modifica la lógica de ingesta de datos (*data intake*) en la función principal del programa para acoplarla de manera exacta a la nueva estructura de carpetas, formatos y archivos indicada por la profesora en `context.md`.

### Tarea 2: Implementación de Casos Extremos de Adyacencia Espacial en TFCE
Necesito evaluar numéricamente el impacto de la topología de la red antes de implementar la métrica avanzada de la bola de radio $R$. Escribe un script o módulo de prueba que reciba una instancia completa combinada del diseño experimental mixto 2x2.

La instancia se compone de los datos cruzados de los siguientes grupos/condiciones:
1. `Full-term (FT) / Silence`
2. `Full-term (FT) / Heartbeat`
3. `Pre-term (PT) / Silence`
4. `Pre-term (PT) / Heartbeat`

Para el cálculo de la extensión del clúster ($e(h, p)$) en la integral continua de TFCE ($\int e(h,p)^E h^H dh$), la adyacencia temporal y espectral es determinista y trivial:
- Una frecuencia $f$ es vecina de $f + 1$ y $f - 1$.
- Una época/ventana $t$ es vecina de $t + 1$ y $t - 1$.

Sin embargo, debes parametrizar y probar **dos criterios de adyacencia espacial (de aristas)** radicalmente opuestos:

#### Caso Extremo A: Adyacencia Total (Grafo Completo)
- **Regla:** Cualquier nodo del espacio de conectividad (es decir, la arista dirigida $(i, j)$) se considera adyacente con **absolutamente todos** los demás nodos/aristas en el espacio del cuero cabelludo.

#### Caso Extremo B: Adyacencia Nula (Grafo Aislado)
- **Regla:** Cada nodo del espacio de conectividad (la arista dirigida $(i, j)$) es vecino **única y exclusivamente de sí mismo**. Ninguna arista comparte vecindad espacial con otra en la red.

---

## 📥 Entregables Esperados

1. **Código modular y optimizado en Python:** Implementación matemática limpia utilizando matrices de `numpy` y algoritmos de etiquetado de componentes conexas eficientes de `scipy` (como `scipy.ndimage.label` o estructuras dispersas) para barrer el dominio de los umbrales $h$ en la suma de Riemann.
2. **Script/Función de Evaluación Comparativa:** Un script que ejecute secuencialmente el pipeline TFCE bajo ambas máscaras de adyacencia (Total vs. Nula) sobre la instancia 2x2 provista, y retorne o grafique los mapas estadísticos resultantes para visualizar cómo cambia la sensibilidad y la amplificación del soporte topológico entre ambos extremos.

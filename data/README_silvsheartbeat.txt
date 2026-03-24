Los datos van en formato EEGLAB.
Cada uno tiene un 
*.set file ->  tiene la estructura del archivo, con metadata
*.fdt file -> tiene los datos numéricos (i.e. la señal, los canales) 

Aquí enviamos los archivos de 1 niño (R004) durante la escucha de latido cardíaco (ne) y durante el silencio (si) 
Son registros continuos con 64 canales , de los cuales hay varios que están en zero (los periféricos) 
Tienen un segundo de baseline (silencio) antes del registro (aunque en el silencio es solo 1 segundo más)

Electrodos:
-foto de los electrodoes (los perifeérics son los que están en el borde) 
son estos: [1 5 10 17 23 29 32 35 37 39 43 47 55 61 62 63 64]
-eeglab_65chanlocs.elp = archivo con las coordenadas de los electrodos (64 + el de rereferencia)


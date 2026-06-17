## Email from prof.


Hola 

En efecto, hay que calcular sobre los periodos de 5 a 10 latidos que estén libres de "ruido".

Por eso, yo les estoy subiendo (va lento):
-epch_silence.zip 
-epch_heartbeat.zip 

aquí: https://drive.google.com/drive/folders/1DGEk_R3DvAdOraQHtVvop286OyAqrwa8?usp=drive_link
En ambos zip los nombres tienen este formato PT_si_obs_101.set ( +  PT_si_obs_101.fdt)  
FT es fullterm, PT es preterm, si es silence y hb es latido cardiaco
obs_XX es el ID número del niño 
 
Topos tienen una estructura 65 channels X 1500 samples X número de épocas sin ruido
Frecuencia de muestreo = 500 hz
Un latido dura 600 ms y cada época o segmento tiene 5 latidos exactos.
5 latidos son los mínimo para calcular 1.67 Hz .con la técnica que se llama " tagging ".
Además, te cuento que los resultados son "mejores" al analizar el tagging al primer armónico 3.33 Hz, quizás porque en latido tiene 2 componentes long-short.   

Ahora, sería genial si pueden calcular la conectividad para estas épocas cortas de 5 latidos , así lo podemos relacionar con los resultados de tagging.
 
Si por el contrario se necesitan periodos más largos, díganme cuantos latidos quieren y les envío las épocas de 10 o más latidos que estén libres de ruido. Con eso vamos a perder niños, pero se puede hacer.
No les subo los continuos porque son "ruidosos", y en un continuo es impreciso eliminar los periodos ruidosos, lo que hay que hacer es segmnetar primero y eliminar después. 

En otra, estos niños hicieron otras tareas  en este estudio, después del silencio y el latido. 
Si les parece les cuento el resto porque estoy justo en la escritura de esos resultados y si se puede enriquecer con esto de la conectividad, sería genial.
 
que tal?
saludos, mp


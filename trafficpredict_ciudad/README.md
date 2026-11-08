# TrafficPredict Ciudad (prototipo)

Conteo de vehículos desde video de cámaras + análisis numérico (Lagrange, Newton, mínimos cuadrados).

## Ejecutar
    pip install -r requirements.txt
    streamlit run app.py

## Archivos
- `app.py`        interfaz Streamlit (pestañas: Cámara y conteo, Datos, Interpolación, Mínimos cuadrados, Ciudad, Acerca)
- `counter.py`    detección (resta de fondo o YOLO), seguimiento (húngaro) y conteo por línea virtual
- `metodos.py`    Lagrange, Newton (diferencias divididas) y mínimos cuadrados desde cero
- `synth_video.py` / `build_demo.py`  generan los 3 videos demo sintéticos y precalculan el conteo
- `videos/`, `salidas/`  videos demo y resultados precalculados

## Usar un video real (carros de lado, en diagonal, etc.)
Sube el video en la barra lateral, ajusta la línea de conteo y pulsa "Procesar video".
Para cámaras reales instala `ultralytics` (YOLO) y elige ese detector; valida contra un conteo manual.

Tipos de línea: Horizontal (carros arriba/abajo), Vertical (carros de izquierda a derecha o al revés)
e Inclinada (vías en diagonal o curva). La vista previa muestra la línea sobre el video antes de procesar.

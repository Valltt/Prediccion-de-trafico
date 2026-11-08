"""Construye los videos de demostración y precalcula el conteo (corre una sola vez)."""
from pathlib import Path
import json
import synth_video as sv
import counter

CAMARAS = {
    "cam1": dict(semilla=11, tasa=1.3, pico=0.55),
    "cam2": dict(semilla=23, tasa=0.8, pico=0.30),
    "cam3": dict(semilla=5, tasa=1.7, pico=0.75),
}
Path("videos").mkdir(exist_ok=True)
Path("salidas").mkdir(exist_ok=True)
LINEA = sv.LINE_Y / sv.H

for nombre, kw in CAMARAS.items():
    ruta = f"videos/{nombre}.mp4"
    real = sv.generar(ruta, segundos=120, **kw)
    clave = counter.clave_cache(nombre, LINEA, "sur-norte", 900, 2200, "fondo")
    df, fps, _ = counter.contar(ruta, f"salidas/{clave}.mp4", linea_frac=LINEA)
    df.to_csv(f"salidas/{clave}.csv", index=False)
    print(nombre, "real", real, "| medido", df.clase.value_counts().to_dict(), len(df))

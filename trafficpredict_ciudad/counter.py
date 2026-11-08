"""Detección, seguimiento y conteo de vehículos sobre video.

Detector por defecto: sustracción de fondo (OpenCV MOG2), que funciona sin
descargar modelos. Si `ultralytics` está instalado se puede usar YOLO
(detector="yolo"), que es lo recomendable con cámaras reales.

Seguimiento: asociación por distancia de centroides.
Conteo: un vehículo se cuenta cuando su centroide cruza la línea virtual en el
sentido configurado (sur-norte = de abajo hacia arriba).
"""
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

CLASES = ["moto", "particular", "pesado"]
COLOR_BGR = {"moto": (0, 140, 255), "particular": (80, 220, 80), "pesado": (230, 120, 40)}

# Mapeo de clases COCO (YOLO) a las categorías del proyecto
COCO_A_CLASE = {"motorcycle": "moto", "bicycle": "moto", "car": "particular",
                "bus": "pesado", "truck": "pesado"}


@dataclass
class Pista:
    id: int
    centro: tuple
    caja: tuple
    area_max: float = 0.0
    perdido: int = 0
    vel: tuple = (0.0, 0.0)
    contado: bool = False
    clase_yolo: str | None = None
    historial: list = field(default_factory=list)


def clasificar_por_area(area, umbral_moto, umbral_pesado):
    if area < umbral_moto:
        return "moto"
    if area < umbral_pesado:
        return "particular"
    return "pesado"


def _ffmpeg():
    """ffmpeg del sistema si existe; si no, el que trae el paquete imageio-ffmpeg (pip)."""
    import shutil
    ruta = shutil.which("ffmpeg")
    if ruta:
        return ruta
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _reencodar(crudo, destino):
    """Convierte a H.264 (reproducible en navegador). Sin ffmpeg deja el mp4 tal cual."""
    exe = _ffmpeg()
    if exe is None:
        Path(crudo).replace(destino)
        return
    subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(crudo), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-preset", "veryfast", str(destino)], check=True)
    Path(crudo).unlink()


def estimar_fondo(video, n=60):
    """Fondo = mediana temporal de n cuadros repartidos por todo el video (cámara fija)."""
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cuadros = []
    for i in np.linspace(0, max(total - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            cuadros.append(fr)
    cap.release()
    return np.median(np.stack(cuadros), axis=0).astype(np.uint8)


class DetectorFondo:
    """Resta de fondo (mediana temporal) + morfología + contornos."""

    def __init__(self, fondo, umbral=14, area_min=120, cierre=(5, 9), apertura=(3, 3)):
        self.fondo = cv2.GaussianBlur(fondo, (5, 5), 0)
        self.umbral = umbral
        self.area_min = area_min
        self.cierre = cv2.getStructuringElement(cv2.MORPH_RECT, cierre)
        self.apertura = cv2.getStructuringElement(cv2.MORPH_RECT, apertura)

    def detectar(self, frame):
        dif = cv2.absdiff(cv2.GaussianBlur(frame, (5, 5), 0), self.fondo).max(axis=2)
        _, mascara = cv2.threshold(dif, self.umbral, 255, cv2.THRESH_BINARY)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, self.apertura)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, self.cierre)
        cnts, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [(*cv2.boundingRect(c), None) for c in cnts if cv2.contourArea(c) >= self.area_min]


class DetectorYolo:
    def __init__(self, modelo="yolov8n.pt", conf=0.30):
        from ultralytics import YOLO  # import perezoso
        self.m = YOLO(modelo)
        self.conf = conf

    def detectar(self, frame):
        r = self.m(frame, verbose=False, conf=self.conf, classes=[2, 3, 5, 7])[0]
        cajas = []
        for b in r.boxes:
            nombre = r.names[int(b.cls)]
            if nombre in COCO_A_CLASE:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                cajas.append((x1, y1, x2 - x1, y2 - y1, COCO_A_CLASE[nombre]))
        return cajas


def contar(video_in, video_out=None, linea_frac=0.55, sentido="sur-norte", detector="fondo", det_obj=None,
           umbral_moto=900, umbral_pesado=2200, dist_max=None, max_perdido=8,
           progreso=None, max_frames=None, linea_der=None, linea=None):
    """Procesa el video y devuelve (DataFrame de eventos, fps, ruta_video_anotado).

    Línea de conteo: o bien `linea=(x1, y1, x2, y2)` en fracciones del cuadro (0-1), o bien horizontal/inclinada
    con linea_frac (borde izquierdo) y linea_der (borde derecho, opcional), como fracción de la altura.
    sentido: "sur-norte" (hacia arriba), "norte-sur" (hacia abajo), "izq-der", "der-izq" o "ambos".
    Para líneas horizontales/inclinadas dibujadas de izquierda a derecha, "sur-norte" = cruzar hacia arriba.
    Para líneas verticales dibujadas de arriba a abajo, "izq-der" = cruzar hacia la derecha.
    """
    cap = cv2.VideoCapture(str(video_in))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total = min(total, max_frames)
    if linea is not None:
        x1, y1, x2, y2 = linea[0] * W, linea[1] * H, linea[2] * W, linea[3] * H
    else:
        x1, y1, x2, y2 = 0.0, H * linea_frac, float(W), H * (linea_der if linea_der is not None else linea_frac)
    largo2 = max((x2 - x1) ** 2 + (y2 - y1) ** 2, 1e-9)
    if dist_max is None:
        dist_max = 0.06 * float(np.hypot(W, H))

    def lado(c):  # signo según el lado de la línea (producto cruz)
        return (x2 - x1) * (c[1] - y1) - (y2 - y1) * (c[0] - x1)

    def sobre_segmento(c):
        t = ((c[0] - x1) * (x2 - x1) + (c[1] - y1) * (y2 - y1)) / largo2
        return 0.0 <= t <= 1.0

    det = det_obj or (DetectorYolo() if detector == "yolo" else DetectorFondo(estimar_fondo(video_in)))
    raw = None
    if video_out:
        raw = Path(video_out).with_suffix(".raw.mp4")
        vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    pistas, siguiente_id, eventos = [], 1, []
    conteo = {c: 0 for c in CLASES}
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and f >= max_frames):
            break
        cajas = det.detectar(frame)

        # --- asociación global (húngaro) con predicción de movimiento
        usadas = set()
        if pistas and cajas:
            cen = np.array([[x + w / 2, y + h / 2] for x, y, w, h, _ in cajas])
            pred = np.array([[p.centro[0] + p.vel[0] * (p.perdido + 1), p.centro[1] + p.vel[1] * (p.perdido + 1)] for p in pistas])
            costo = np.linalg.norm(pred[:, None, :] - cen[None, :, :], axis=2)
            costo_g = np.where(costo > dist_max, 1e6, costo)
            filas, cols = linear_sum_assignment(costo_g)
            asign = {r: c for r, c in zip(filas, cols) if costo_g[r, c] < 1e6}
        else:
            asign = {}
        for k, p in enumerate(pistas):
            if k not in asign:
                p.perdido += 1
                continue
            mejor = asign[k]
            usadas.add(mejor)
            x, y, w, h, cl = cajas[mejor]
            previo = p.centro
            nuevo_c = (x + w / 2, y + h / 2)
            p.vel = ((nuevo_c[0] - previo[0]) / (p.perdido + 1), (nuevo_c[1] - previo[1]) / (p.perdido + 1))
            p.centro = nuevo_c
            p.caja = (x, y, w, h)
            p.perdido = 0
            p.area_max = max(p.area_max, w * h)
            if cl:
                p.clase_yolo = cl
            l0, l1 = lado(previo), lado(p.centro)
            hacia_a = l0 > 0 >= l1          # sur-norte / izq-der
            hacia_b = l0 < 0 <= l1          # norte-sur / der-izq
            cruza = sobre_segmento(p.centro) and (
                (hacia_a and sentido in ("sur-norte", "izq-der", "ambos")) or
                (hacia_b and sentido in ("norte-sur", "der-izq", "ambos")))
            if cruza and not p.contado:
                p.contado = True
                clase = p.clase_yolo or clasificar_por_area(p.area_max, umbral_moto, umbral_pesado)
                conteo[clase] += 1
                eventos.append(dict(id=p.id, cuadro=f, segundo=f / fps, clase=clase, area=int(p.area_max),
                                    sentido="A" if hacia_a else "B"))
        for i, (x, y, w, h, cl) in enumerate(cajas):
            if i not in usadas:
                pistas.append(Pista(id=siguiente_id, centro=(x + w / 2, y + h / 2), caja=(x, y, w, h),
                                    area_max=w * h, clase_yolo=cl))
                siguiente_id += 1
        pistas = [p for p in pistas if p.perdido <= max_perdido]

        # --- anotación
        if raw is not None:
            vis = frame.copy()
            cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            for p in pistas:
                if p.perdido:
                    continue
                x, y, w, h = p.caja
                clase = p.clase_yolo or clasificar_por_area(p.area_max, umbral_moto, umbral_pesado)
                col = COLOR_BGR[clase]
                cv2.rectangle(vis, (x, y), (x + w, y + h), col, 2)
                cv2.putText(vis, f"{clase} #{p.id}", (x, max(y - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
            cv2.rectangle(vis, (0, 0), (190, 78), (20, 20, 20), -1)
            cv2.putText(vis, f"Total: {sum(conteo.values())}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            for k, c in enumerate(CLASES):
                cv2.putText(vis, f"{c}: {conteo[c]}", (8, 36 + 14 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_BGR[c], 1)
            vw.write(vis)
        f += 1
        if progreso and total and f % 10 == 0:
            progreso(min(f / total, 1.0))
    cap.release()

    salida = None
    if raw is not None:
        vw.release()
        _reencodar(raw, video_out)
        salida = str(video_out)
    if progreso:
        progreso(1.0)
    df = pd.DataFrame(eventos, columns=["id", "cuadro", "segundo", "clase", "area", "sentido"])
    return df, fps, salida


def por_intervalos(df, duracion_s, intervalo_s):
    """Agrupa eventos en intervalos de `intervalo_s` segundos -> tabla estilo aforo."""
    n = int(np.ceil(duracion_s / intervalo_s))
    filas = []
    for i in range(n):
        a, b = i * intervalo_s, (i + 1) * intervalo_s
        sub = df[(df.segundo >= a) & (df.segundo < b)]
        fila = {"inicio_s": a, "fin_s": b}
        for c in CLASES:
            fila[c] = int((sub.clase == c).sum())
        fila["total_vehiculos"] = int(len(sub))
        filas.append(fila)
    return pd.DataFrame(filas)


def clave_cache(nombre, linea_frac, sentido, umbral_moto, umbral_pesado, detector, linea_der=None, linea=None):
    """Nombre de archivo estable para guardar el resultado de un procesamiento."""
    der = "" if linea_der is None or abs(linea_der - linea_frac) < 1e-6 else f"-{linea_der:.3f}"
    geom = "" if linea is None else "_G" + "-".join(f"{v:.3f}" for v in linea)
    return f"{nombre}_L{linea_frac:.3f}{der}{geom}_{sentido}_m{umbral_moto}_p{umbral_pesado}_{detector}"


def dibujar_linea_previa(video, linea, ancho_max=900):
    """Primer cuadro del video con la línea de conteo dibujada (RGB) para vista previa."""
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(n // 3, 0))
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return None
    H, W = fr.shape[:2]
    x1, y1, x2, y2 = linea
    cv2.line(fr, (int(x1 * W), int(y1 * H)), (int(x2 * W), int(y2 * H)), (0, 0, 255), 3)
    cv2.circle(fr, (int(x1 * W), int(y1 * H)), 6, (0, 255, 255), -1)
    cv2.circle(fr, (int(x2 * W), int(y2 * H)), 6, (0, 255, 255), -1)
    return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)

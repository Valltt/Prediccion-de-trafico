"""TrafficPredict Ciudad: conteo vehicular desde cámaras + análisis numérico."""
import importlib.util
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import counter
import metodos

BASE = Path(__file__).parent
CLASES = counter.CLASES
NOMBRES = {"moto": "Motos", "particular": "Particulares", "pesado": "Buses y camiones"}
CAMARAS = {
    "Cámara 1 · Corredor norte (demo)": "cam1",
    "Cámara 2 · Corredor oriental (demo)": "cam2",
    "Cámara 3 · Corredor central (demo)": "cam3",
}
LINEA_DEFECTO, UMB_MOTO, UMB_PESADO = 0.556, 900, 2200
YOLO_OK = importlib.util.find_spec("ultralytics") is not None

st.set_page_config(page_title="TrafficPredict Ciudad", page_icon="🚦", layout="wide")


# ---------------------------------------------------------------- utilidades
def mmss(seg):
    seg = int(round(seg))
    return f"{seg // 60:02d}:{seg % 60:02d}"


def info_video(ruta):
    cap = cv2.VideoCapture(str(ruta))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n / fps, fps


def cargar_verdad(ruta):
    p = Path(ruta).with_suffix(".verdad.json")
    return json.load(open(p)) if p.exists() else None


def figura():
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.grid(alpha=0.3)
    return fig, ax


# ---------------------------------------------------------------- barra lateral
with st.sidebar:
    st.header("Fuente de video")
    camara = st.selectbox("Cámara", list(CAMARAS))
    subido = st.file_uploader("Cargar otro video (opcional)", type=["mp4", "avi", "mov", "mkv"])

    st.header("Configuración")
    intervalo = st.selectbox("Duración del intervalo (s)", [5, 10, 15, 20, 30], index=1)
    tipo_linea = st.radio("Tipo de línea de conteo", ["Horizontal", "Vertical", "Inclinada"], horizontal=True,
                          help="Horizontal: carros que van arriba/abajo. Vertical: carros que van de lado "
                               "(izquierda/derecha). Inclinada: vías en diagonal o en curva.")
    linea, linea_der, geom = LINEA_DEFECTO, None, None
    if tipo_linea == "Horizontal":
        linea = st.slider("Posición de la línea (desde arriba)", 0.05, 0.95, LINEA_DEFECTO, 0.001, format="%.3f")
        opciones_sentido = ["sur-norte", "norte-sur", "ambos"]
        ayuda_sentido = "sur-norte: cruzan hacia arriba del cuadro. norte-sur: hacia abajo."
    elif tipo_linea == "Vertical":
        xv = st.slider("Posición de la línea (desde la izquierda)", 0.05, 0.95, 0.5, 0.001, format="%.3f")
        linea, geom = 0.0, (xv, 0.0, xv, 1.0)
        opciones_sentido = ["izq-der", "der-izq", "ambos"]
        ayuda_sentido = "izq-der: cruzan hacia la derecha. der-izq: hacia la izquierda."
    else:
        st.caption("Dibuja la línea de un punto A a un punto B (fracciones del cuadro).")
        ca, cb = st.columns(2)
        x1 = ca.slider("A · x", 0.0, 1.0, 0.0, 0.01)
        y1 = cb.slider("A · y", 0.0, 1.0, 0.45, 0.01)
        x2 = ca.slider("B · x", 0.0, 1.0, 1.0, 0.01)
        y2 = cb.slider("B · y", 0.0, 1.0, 0.75, 0.01)
        linea, geom = 0.0, (x1, y1, x2, y2)
        opciones_sentido = ["sur-norte", "norte-sur", "ambos"]
        ayuda_sentido = ("Con A a la izquierda de B: sur-norte cruza hacia arriba de la línea y norte-sur hacia abajo. "
                         "Si no estás seguro, elige 'ambos'.")
    sentido = st.radio("Sentido a contar", opciones_sentido, horizontal=True, help=ayuda_sentido)
    opciones_det = ["Resta de fondo"] + (["YOLO (ultralytics)"] if YOLO_OK else [])
    detector_txt = st.selectbox("Detector", opciones_det, index=len(opciones_det) - 1 if subido is not None else 0,
                                help="Resta de fondo: cámara fija vista desde arriba. "
                                     "YOLO: recomendado para videos con perspectiva, sombras o cámara que se mueve.")
    if subido is not None and not YOLO_OK:
        st.caption("Para videos reales instala YOLO: `python -m pip install ultralytics`")
    detector = "yolo" if detector_txt.startswith("YOLO") else "fondo"
    with st.expander("Umbrales de clasificación por área"):
        umb_moto = st.number_input("Moto si área < (px²)", 100, 5000, UMB_MOTO, 50)
        umb_pesado = st.number_input("Pesado si área ≥ (px²)", 500, 20000, UMB_PESADO, 100)
        if detector == "yolo":
            st.caption("Con YOLO la clase viene del modelo y estos umbrales no se usan.")

# --------------------------------------------------------------- video elegido
if subido is not None:
    carpeta = BASE / "subidos"
    carpeta.mkdir(exist_ok=True)
    ruta_video = carpeta / subido.name
    ruta_video.write_bytes(subido.getbuffer())
    nombre = "up_" + ruta_video.stem
    etiqueta = f"Video cargado · {subido.name}"
else:
    nombre = CAMARAS[camara]
    ruta_video = BASE / "videos" / f"{nombre}.mp4"
    etiqueta = camara

clave = counter.clave_cache(nombre, linea, sentido, int(umb_moto), int(umb_pesado), detector, linea_der, geom)
(BASE / "salidas").mkdir(exist_ok=True)
ruta_csv = BASE / "salidas" / f"{clave}.csv"
ruta_anot = BASE / "salidas" / f"{clave}.mp4"
duracion, fps = info_video(ruta_video)

st.title("TrafficPredict Ciudad")
st.caption(f"Conteo vehicular con visión por computador · {etiqueta} · Sentido {sentido.replace('-', '–')}")

procesado = ruta_csv.exists() and ruta_anot.exists()
if not procesado:
    st.info("Estos parámetros aún no se han procesado sobre este video. "
            "Pulsa el botón para detectar, seguir y contar los vehículos.")
    if st.button("Procesar video", type="primary"):
        barra = st.progress(0.0, text="Detectando y contando vehículos…")
        try:
            df_ev, _, _ = counter.contar(ruta_video, ruta_anot, linea_frac=linea, linea_der=linea_der, linea=geom, sentido=sentido,
                                         detector=detector, umbral_moto=int(umb_moto),
                                         umbral_pesado=int(umb_pesado),
                                         progreso=lambda p: barra.progress(p, text="Detectando y contando vehículos…"))
        except Exception as e:  # p. ej. modelo YOLO no disponible sin internet
            barra.empty()
            st.error(f"No se pudo procesar el video: {e}")
            st.stop()
        df_ev.to_csv(ruta_csv, index=False)
        st.rerun()
    prev = counter.dibujar_linea_previa(ruta_video, geom if geom else (0, linea, 1, linea))
    if prev is not None:
        st.subheader("Vista previa de la línea de conteo")
        st.image(prev, caption="Ajusta la posición en la barra lateral hasta que la línea cruce la vía.")
    st.subheader("Video original")
    st.video(str(ruta_video))
    st.stop()

eventos = pd.read_csv(ruta_csv)
tabla = counter.por_intervalos(eventos, duracion, intervalo)
tabla.insert(0, "hora_inicio", tabla.inicio_s.map(mmss))
tabla.insert(1, "hora_fin", tabla.fin_s.map(mmss))
N = len(tabla)
i_max = int(tabla.total_vehiculos.idxmax())

c1, c2, c3 = st.columns(3)
c1.metric("Intervalos disponibles", N)
c2.metric("Vehículos registrados", f"{len(eventos):,}".replace(",", "."))
c3.metric(f"Mayor aforo en {intervalo} s", int(tabla.total_vehiculos.max()),
          delta=tabla.hora_inicio[i_max], delta_color="normal")

tab_cam, tab_datos, tab_interp, tab_mc, tab_ciudad, tab_acerca = st.tabs(
    ["Cámara y conteo", "Datos", "Interpolación", "Mínimos cuadrados", "Ciudad", "Acerca del proyecto"])

# ------------------------------------------------------------ Cámara y conteo
with tab_cam:
    st.header("Detección y conteo en video")
    a, b = st.columns(2)
    with a:
        st.subheader("Video original")
        st.video(str(ruta_video))
    with b:
        st.subheader("Video con detección")
        st.video(str(ruta_anot))
    st.caption("La línea roja es la línea virtual de conteo. Cada vehículo se cuenta una sola vez, "
               "cuando su centro la cruza en el sentido elegido.")

    cols = st.columns(len(CLASES) + 1)
    for col, c in zip(cols, CLASES):
        col.metric(NOMBRES[c], int((eventos.clase == c).sum()))
    cols[-1].metric("Flujo medio", f"{len(eventos) / duracion * 3600:,.0f} veh/h".replace(",", "."))

    verdad = cargar_verdad(ruta_video)
    if verdad and subido is None:
        st.subheader("Validación contra el conteo real")
        filas = []
        for c in CLASES:
            real, det = verdad["por_clase"][c], int((eventos.clase == c).sum())
            filas.append({"clase": NOMBRES[c], "real": real, "detectado": det, "diferencia": det - real})
        filas.append({"clase": "Total", "real": verdad["total"], "detectado": len(eventos),
                      "diferencia": len(eventos) - verdad["total"]})
        st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")
        st.caption("El video demo es sintético, por eso se conoce el conteo real. "
                   "Con cámaras reales hay que validar contra un conteo manual.")

    st.subheader("Vehículos acumulados")
    fig, ax = figura()
    for c in CLASES:
        t = np.sort(eventos[eventos.clase == c].segundo.values)
        ax.step(np.r_[0, t, duracion], np.r_[0, np.arange(1, len(t) + 1), len(t)], where="post", label=NOMBRES[c])
    ax.set_xlabel("Segundos de video")
    ax.set_ylabel("Vehículos acumulados")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

# ----------------------------------------------------------------------- Datos
with tab_datos:
    st.header("Registros de aforo")
    mostrar = tabla[["hora_inicio", "hora_fin", *CLASES, "total_vehiculos"]]
    st.dataframe(mostrar, hide_index=True, width="stretch")
    st.download_button("Descargar CSV", mostrar.to_csv(index=False).encode(), f"aforo_{nombre}.csv", "text/csv")
    fig, ax = figura()
    ax.plot(tabla.inicio_s, tabla.total_vehiculos, marker="o")
    ax.set_xticks(tabla.inicio_s)
    ax.set_xticklabels(tabla.hora_inicio, rotation=45)
    ax.set_title(f"Flujo vehicular por intervalo de {intervalo} segundos")
    ax.set_xlabel("Hora de inicio (mm:ss del video)")
    ax.set_ylabel("Vehículos")
    st.pyplot(fig)
    plt.close(fig)

xs_all = tabla.inicio_s.values.astype(float)
ys_all = tabla.total_vehiculos.values.astype(float)

# ---------------------------------------------------------------- Interpolación
with tab_interp:
    st.header("Interpolación polinómica")
    st.write(f"Selecciona puntos de la serie. La variable x son los segundos desde el inicio del video "
             f"e y los vehículos contados en cada {intervalo} s.")
    if N < 3:
        st.warning("Se necesitan al menos 3 intervalos. Reduce la duración del intervalo.")
    else:
        metodo = st.radio("Método numérico", ["Lagrange", "Newton"], horizontal=True)
        n_pts = st.slider("Cantidad de puntos", 3, min(8, N), min(4, N))
        idx = np.unique(np.linspace(0, N - 1, n_pts).round().astype(int))
        xs, ys = xs_all[idx], ys_all[idx]
        st.write("Puntos utilizados:")
        st.dataframe(pd.DataFrame({"Hora": tabla.hora_inicio.values[idx], "Vehículos": ys.astype(int)}),
                     hide_index=True, width="stretch")
        xq = st.slider("Segundo para interpolar", int(xs.min()), int(xs.max()), int((xs.min() + xs.max()) / 2))

        if metodo == "Lagrange":
            st.latex(r"P(x)=\sum_{i=0}^{n} y_i \prod_{j\neq i}\frac{x-x_j}{x_i-x_j}")
            f = lambda x: metodos.lagrange(xs, ys, x)
        else:
            st.latex(r"P(x)=a_0+a_1(x-x_0)+a_2(x-x_0)(x-x_1)+\cdots")
            _, coef, T = metodos.newton(xs, ys, np.array([xq], float))
            st.write("Tabla de diferencias divididas:")
            st.dataframe(pd.DataFrame(T).round(4), width="stretch")
            st.write("Coeficientes de Newton:")
            st.dataframe(pd.DataFrame({"a_k": coef}).round(4), width="stretch")
            f = lambda x: metodos.newton(xs, ys, x)[0]

        valor = float(f(np.array([xq], float))[0])
        m1, m2 = st.columns(2)
        m1.metric("Tiempo consultado", mmss(xq))
        m2.metric("Valor interpolado", f"{valor:.2f} veh.")
        st.caption("La interpolación estima valores entre los puntos seleccionados. "
                   "No constituye por sí sola una predicción del tráfico futuro.")

        g = np.linspace(xs.min(), xs.max(), 300)
        fig, ax = figura()
        ax.scatter(xs, ys, label="Puntos seleccionados", zorder=3)
        ax.plot(g, f(g), label=f"Polinomio de {metodo}")
        ax.scatter([xq], [valor], marker="x", s=90, color="tab:orange", label="Consulta", zorder=4)
        otros = np.setdiff1d(np.arange(N), idx)
        if len(otros):
            ax.scatter(xs_all[otros], ys_all[otros], facecolors="none", edgecolors="gray", label="Intervalos no usados")
        ax.set_title(f"Interpolación por {metodo}")
        ax.set_xlabel("Segundos desde el inicio del video")
        ax.set_ylabel(f"Vehículos por {intervalo} s")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        dentro = [i for i in otros if xs.min() <= xs_all[i] <= xs.max()]
        if dentro:
            err = np.abs(f(xs_all[dentro]) - ys_all[dentro])
            st.subheader("Error en intervalos no utilizados")
            st.dataframe(pd.DataFrame({"Hora": tabla.hora_inicio.values[dentro], "Observado": ys_all[dentro].astype(int),
                                       "Interpolado": f(xs_all[dentro]).round(1), "Error abs.": err.round(1)}),
                         hide_index=True, width="stretch")
            st.caption(f"Error absoluto medio en intervalos retenidos: {err.mean():.2f} vehículos. "
                       "Estos intervalos no entraron al polinomio, así que es una validación real.")

# ------------------------------------------------------------ Mínimos cuadrados
with tab_mc:
    st.header("Ajuste por mínimos cuadrados")
    st.write("Este ajuste aproxima la tendencia de los datos; no interpola necesariamente cada punto.")
    grado = st.selectbox("Grado del polinomio de ajuste", list(range(1, min(5, N - 1) + 1)), index=min(1, N - 2))
    coef, pred, mae, rmse = metodos.minimos_cuadrados(xs_all, ys_all, grado)
    k1, k2, k3 = st.columns(3)
    k1.metric("MAE", f"{mae:.2f} veh.")
    k2.metric("RMSE", f"{rmse:.2f} veh.")
    k3.metric("Grado", grado)
    st.write("Coeficientes (de mayor a menor potencia, con x centrado en su media):")
    st.code(np.array2string(coef, precision=4))
    g = np.linspace(xs_all.min(), xs_all.max(), 300)
    fig, ax = figura()
    ax.scatter(xs_all, ys_all, label="Datos observados")
    ax.plot(g, pred(g), label="Ajuste por mínimos cuadrados")
    ax.set_title("Ajuste polinómico")
    ax.set_xlabel("Segundos desde el inicio del video")
    ax.set_ylabel(f"Vehículos por {intervalo} s")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("MAE y RMSE se calculan sobre los mismos datos usados para ajustar el polinomio; "
               "son errores de ajuste, no una validación independiente.")

# ----------------------------------------------------------------------- Ciudad
with tab_ciudad:
    st.header("Resumen por cámara")
    filas = []
    for etq, cid in CAMARAS.items():
        k = counter.clave_cache(cid, LINEA_DEFECTO, "sur-norte", UMB_MOTO, UMB_PESADO, "fondo")
        p = BASE / "salidas" / f"{k}.csv"
        if p.exists():
            d = pd.read_csv(p)
            fila = {"Cámara": etq}
            fila.update({NOMBRES[c]: int((d.clase == c).sum()) for c in CLASES})
            fila["Total"] = len(d)
            fila["Flujo (veh/h)"] = round(len(d) / 120 * 3600)
            filas.append(fila)
    if filas:
        res = pd.DataFrame(filas)
        st.dataframe(res, hide_index=True, width="stretch")
        fig, ax = figura()
        base = np.zeros(len(res))
        for c in CLASES:
            ax.bar(res["Cámara"].str.slice(0, 8), res[NOMBRES[c]], bottom=base, label=NOMBRES[c])
            base += res[NOMBRES[c]].values
        ax.set_ylabel("Vehículos en 120 s de video")
        ax.set_title("Composición del flujo por cámara")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Resultados con los parámetros por defecto. Las cámaras demo son videos sintéticos.")
    else:
        st.info("Procesa las cámaras con los parámetros por defecto para ver el resumen.")

# ----------------------------------------------------------------------- Acerca
with tab_acerca:
    st.header("Objetivo")
    st.write("Desarrollar una herramienta académica que cuenta vehículos a partir de video de cámaras de tráfico "
             "y analiza el flujo resultante con interpolación de Lagrange, Newton y ajuste por mínimos cuadrados.")
    st.subheader("Cómo funciona el conteo")
    st.markdown(
        "1. **Detección:** se estima el fondo de la escena (mediana temporal) y se restan los cuadros para aislar los vehículos. "
        "Con cámaras reales se puede cambiar a YOLO.\n"
        "2. **Seguimiento:** cada vehículo recibe un identificador y se asocia entre cuadros con asignación óptima "
        "(algoritmo húngaro) y predicción de movimiento.\n"
        "3. **Conteo:** un vehículo se cuenta una vez, cuando su centro cruza la línea virtual en el sentido elegido.\n"
        "4. **Clasificación:** por tamaño (motos, particulares, buses y camiones), o por clase del modelo con YOLO.\n"
        "5. **Agregación:** los conteos se agrupan por intervalos y alimentan los métodos numéricos.")
    st.markdown("**Alcance de los datos demo:** tres videos sintéticos de 120 s con conteo real conocido, "
                "creados para comprobar que el sistema cuenta bien. No son cámaras reales ni monitoreo en tiempo real.")
    st.warning("Limitaciones: la resta de fondo supone cámara fija e iluminación estable, y puede fallar con sombras, "
               "lluvia u oclusiones. Los umbrales de tamaño dependen del encuadre de cada cámara. "
               "Para cámaras reales se recomienda YOLO y validar contra un conteo manual.")
    st.info("Para hacer predicciones del tráfico futuro será necesario acumular aforos de varias jornadas "
            "y separar datos de entrenamiento y validación.")

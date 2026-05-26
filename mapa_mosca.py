# ============================================================
# MAPA EPIDEMIOLÓGICO AGRÍCOLA — Mosca de la Fruta
# v5 — Performance fixes aplicados:
#   1. load_polygons_from_kml con @st.cache_data (evita re-parseo del KMZ)
#   2. KMZ se lee como bytes antes de pasar a la función cacheada
#   3. Filtro dentro del KMZ con bbox pre-filter + shapely vectorizado
#   4. GRID_RES reducido a 200 (40k pts vs 360k) + contains_xy vectorizado
#   5. st_folium con key="main_map" para evitar re-renderizados innecesarios
# ============================================================

# --- IMPORTS ---
import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium import FeatureGroup
from streamlit_folium import st_folium
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from shapely.prepared import prep
from pykml import parser
import zipfile, io, re, math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import base64
import tempfile
import os
from html import escape
from PIL import Image as PILImage

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

# ============================================================
# CONFIGURACIÓN PÁGINA
# ============================================================
st.set_page_config(
    page_title="Mapa Epidemiológico Agrícola — Mosca de la Fruta",
    layout="wide"
)

st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 0rem; }
    #MainMenu, footer { visibility: hidden; }
    @keyframes fly {
        0%   { transform: translate(0px,  0px) rotate(0deg)   scale(1);    }
        15%  { transform: translate(3px, -4px) rotate(15deg)  scale(1.1);  }
        30%  { transform: translate(-2px,-7px) rotate(-10deg) scale(0.95); }
        45%  { transform: translate(5px, -3px) rotate(20deg)  scale(1.05); }
        60%  { transform: translate(-3px,-5px) rotate(-15deg) scale(1);    }
        75%  { transform: translate(2px, -2px) rotate(10deg)  scale(1.08); }
        100% { transform: translate(0px,  0px) rotate(0deg)   scale(1);    }
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# FUNCIONES KMZ / KML
# ============================================================
def extract_kml_from_kmz(kmz_path: str) -> bytes:
    with zipfile.ZipFile(kmz_path, "r") as z:
        kml_names = [n for n in z.namelist() if n.lower().endswith(".kml")]
        if not kml_names:
            raise FileNotFoundError("No se encontró .kml dentro del KMZ.")
        main = next(
            (n for n in kml_names if n.split("/")[-1].lower() == "doc.kml"),
            kml_names[0]
        )
        return z.read(main)


def extract_kml_from_kmz_bytes(kmz_bytes: bytes) -> bytes:
    """Igual que extract_kml_from_kmz pero recibe bytes directamente (para cacheo)."""
    with zipfile.ZipFile(io.BytesIO(kmz_bytes), "r") as z:
        kml_names = [n for n in z.namelist() if n.lower().endswith(".kml")]
        if not kml_names:
            raise FileNotFoundError("No se encontró .kml dentro del KMZ.")
        main = next(
            (n for n in kml_names if n.split("/")[-1].lower() == "doc.kml"),
            kml_names[0]
        )
        return z.read(main)


def parse_coordinates_kml(coord_text: str):
    coords = []
    if not coord_text:
        return coords
    coord_text = re.sub(r"\s+", " ", coord_text.strip())
    for triplet in coord_text.split(" "):
        if not triplet:
            continue
        parts = triplet.split(",")
        if len(parts) >= 2:
            coords.append((float(parts[1]), float(parts[0])))
    return coords


# ============================================================
# FIX 1: @st.cache_data en load_polygons_from_kml
# Recibe bytes (hasheable) en lugar de path/BytesIO
# ============================================================
@st.cache_data(show_spinner="Leyendo KMZ…")
def load_polygons_from_kml(kml_bytes: bytes):
    import re as _re

    kml_str = kml_bytes.decode("utf-8", errors="replace")

    if 'xmlns:xsi' not in kml_str:
        kml_str = _re.sub(
            r'(<kml\b)',
            r'\1 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
            kml_str,
            count=1
        )

    if 'xmlns:xsi' not in kml_str:
        kml_str = _re.sub(
            r'(<\w+\b)',
            r'\1 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
            kml_str,
            count=1
        )

    kml_bytes_clean = kml_str.encode("utf-8")

    try:
        root = parser.fromstring(kml_bytes_clean)
    except Exception as e1:
        try:
            from lxml import etree
            root = etree.fromstring(
                kml_bytes_clean,
                parser=etree.XMLParser(recover=True)
            )
        except Exception as e2:
            raise RuntimeError(f"No se pudo parsear el KML: {e1} | {e2}")

    ns = "{http://www.opengis.net/kml/2.2}"
    polygons = []
    for pm in root.findall(f".//{ns}Placemark"):
        name_el = pm.find(f"{ns}name")
        pname   = name_el.text if name_el is not None else "Polígono"
        for poly in pm.findall(f".//{ns}Polygon"):
            outer = poly.find(
                f".//{ns}outerBoundaryIs/{ns}LinearRing/{ns}coordinates"
            )
            if outer is None or not outer.text:
                continue
            folium_coords = parse_coordinates_kml(outer.text)
            if len(folium_coords) < 3:
                continue
            lonlat = [(lon, lat) for (lat, lon) in folium_coords]
            try:
                shp = Polygon(lonlat)
                if shp.is_valid and not shp.is_empty:
                    polygons.append({
                        "name":             pname,
                        "folium_coords":    folium_coords,
                        "shapely_polygon":  shp,
                    })
            except Exception:
                continue
    return polygons


# ============================================================
# CONVERSIÓN DMS → DECIMAL
# ============================================================
def dms_to_decimal(coord_str):
    if pd.isna(coord_str):
        return np.nan
    try:
        return float(coord_str)
    except (ValueError, TypeError):
        pass
    coord_str = str(coord_str).strip()
    match = re.search(r"(\d+)°\s*(\d+)'\s*([\d.]+)\s*([NSEO])", coord_str)
    if match:
        dec = float(match.group(1)) + float(match.group(2))/60 + float(match.group(3))/3600
        if match.group(4) in ["S", "O", "W"]:
            dec = -dec
        return dec
    try:
        return float(coord_str)
    except Exception:
        return np.nan


# ============================================================
# CARGA DE DATOS
# ============================================================
@st.cache_data(show_spinner="Cargando datos de trampas…")
def load_trampas_anexadas() -> pd.DataFrame:
    path_aquai  = "data/BD_Mosca_Fruta_AQUAI.xlsx"
    path_aquaii = "data/BD_Mosca_Fruta_AQUAII.xlsx"

    df1 = pd.read_excel(path_aquai,  sheet_name="Bdatos")
    df2 = pd.read_excel(path_aquaii, sheet_name="BDatos AQU II")
    df  = pd.concat([df1, df2], ignore_index=True)

    rename_map = {
        "LATITUD": "lat", "LONGITUD": "lon", "FECHA": "fecha",
        "FUNDO": "fundo", "MODULO": "modulo", "TURNO": "turno",
        "TRAMPA": "trampa", "CAPTURAS": "capturas", "LOTE": "lote",
        "EMPRESA": "empresa", "SEMANA": "semana", "AÑO": "anio",
        "TIPO DE TRAMPA": "tipo_trampa",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    df["lat"]      = df["lat"].apply(dms_to_decimal)
    df["lon"]      = df["lon"].apply(dms_to_decimal)
    df["capturas"] = pd.to_numeric(df["capturas"], errors="coerce").fillna(0)
    df["fecha"]    = pd.to_datetime(df["fecha"], dayfirst=True, errors="coerce").dt.date
    df["anio"]     = pd.to_numeric(df["anio"],   errors="coerce")
    df["semana"]   = pd.to_numeric(df["semana"], errors="coerce")

    df = df.dropna(subset=["lat", "lon"])
    df = df[(df["lat"].between(-90, 90)) & (df["lon"].between(-180, 180))]

    for c in ["empresa", "fundo", "modulo", "turno", "trampa", "lote", "tipo_trampa"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    return df


# ============================================================
# FIX 2: Filtro dentro del KMZ vectorizado
# Bbox pre-filter + shapely vectorizado (shapely>=2) o loop
# solo sobre filas que pasan el bbox
# ============================================================
def filter_within_union(df: pd.DataFrame, union_poly) -> pd.DataFrame:
    if union_poly is None or df.empty:
        return df

    # Paso 1: bbox rápido (descarta la mayoría fuera del área)
    minx, miny, maxx, maxy = union_poly.bounds
    mask_bbox = (
        df["lon"].between(minx, maxx) &
        df["lat"].between(miny, maxy)
    )
    df_bbox = df[mask_bbox].copy()
    if df_bbox.empty:
        return df_bbox

    # Paso 2: contención exacta
    try:
        # shapely >= 2.0 — completamente vectorizado, sin loop Python
        from shapely import contains_xy
        mask_exact = contains_xy(union_poly,
                                  df_bbox["lon"].values,
                                  df_bbox["lat"].values)
    except ImportError:
        # shapely < 2.0 — loop Python pero solo sobre filas del bbox
        _prep = prep(union_poly)
        mask_exact = [
            _prep.contains(Point(row.lon, row.lat))
            for row in df_bbox.itertuples()
        ]

    return df_bbox[mask_exact].copy()


# ============================================================
# PALETA SEMAFÓRICA
# ============================================================
SEMAFORO_COLORS = {0: "#00FF00", 1: "#FFFF00", 2: "#FFA500", 3: "#FF0000"}

def get_semaforo_category(val: float) -> int:
    v = float(val)
    if v <= 0:    return 0
    elif v < 1.5: return 1
    elif v < 2.5: return 2
    else:         return 3

def get_color_normal(val: float) -> str:
    return SEMAFORO_COLORS[get_semaforo_category(val)]

def get_color_espectral(val: float) -> str:
    verde    = np.array([0,   255,   0], dtype=float)
    amarillo = np.array([255, 255,   0], dtype=float)
    naranja  = np.array([255, 165,   0], dtype=float)
    rojo     = np.array([255,   0,   0], dtype=float)
    t = min(max(float(val), 0.0), 3.0)
    if t <= 1.0:   color = verde    + (amarillo - verde)    * t
    elif t <= 2.0: color = amarillo + (naranja  - amarillo) * (t - 1.0)
    else:          color = naranja  + (rojo     - naranja)  * (t - 2.0)
    return f"#{int(color[0]):02x}{int(color[1]):02x}{int(color[2]):02x}"


# ============================================================
# SAETA (FLECHA CON PUNTA TRIANGULAR)
# ============================================================
def draw_arrow(fg, lat0, lon0, lat1, lon1, color, weight, opacity,
               head_size_deg=0.00006, tooltip=""):
    folium.PolyLine(
        locations=[(lat0, lon0), (lat1, lon1)],
        color=color, weight=weight, opacity=opacity,
        tooltip=tooltip
    ).add_to(fg)

    dlat  = lat1 - lat0
    dlon  = lon1 - lon0
    angle = math.atan2(dlat, dlon)
    hs    = head_size_deg * (0.7 + weight * 0.15)

    tip_lat = lat1
    tip_lon = lon1

    left_angle  = angle + math.radians(155)
    right_angle = angle - math.radians(155)

    left_lat  = lat1 + hs * math.sin(left_angle)
    left_lon  = lon1 + hs * math.cos(left_angle)
    right_lat = lat1 + hs * math.sin(right_angle)
    right_lon = lon1 + hs * math.cos(right_angle)

    folium.Polygon(
        locations=[(tip_lat, tip_lon), (left_lat, left_lon), (right_lat, right_lon)],
        color=color, fill=True, fill_color=color,
        fill_opacity=opacity, weight=0, tooltip=tooltip
    ).add_to(fg)


# ============================================================
# GRADIENTE ESPACIAL — VECTORES DE PROPAGACIÓN
# ============================================================
def compute_gradient_vectors(grid_x, grid_y, gz_smooth, mask_poly,
                              n_arrows=15, min_magnitude=0.05):
    gz = gz_smooth.copy()
    gz[~mask_poly] = np.nan
    grad_x, grad_y = np.gradient(gz)

    nx, ny  = gz.shape
    step_x  = max(1, nx // n_arrows)
    step_y  = max(1, ny // n_arrows)
    vectors = []

    for i in range(step_x // 2, nx, step_x):
        for j in range(step_y // 2, ny, step_y):
            if not mask_poly[i, j]:
                continue
            val = gz[i, j]
            gx  = grad_x[i, j]
            gy  = grad_y[i, j]
            if not (math.isfinite(val) and math.isfinite(gx) and math.isfinite(gy)):
                continue
            mag = math.sqrt(gx**2 + gy**2)
            if mag < min_magnitude or not math.isfinite(mag) or mag == 0.0:
                continue
            vectors.append({
                "lat":       float(grid_y[i, j]),
                "lon":       float(grid_x[i, j]),
                "dlat":      float(gy / mag),
                "dlon":      float(gx / mag),
                "magnitude": float(mag),
                "density":   float(val),
            })
    return vectors


# ============================================================
# CARGA INICIAL — KMZ desde GitHub (cacheable)
# Descarga el KMZ directo del repo via GitHub Contents API.
# El token viene de .streamlit/secrets.toml
# ============================================================
GITHUB_TOKEN_KMZ = st.secrets.get("GITHUB_TOKEN_KMZ", "")
KMZ_API_URL = (
    "https://api.github.com/repos/"
    "controloperacionalprize-boss/CAMPO_RENDIMIENTO/"
    "contents/MODULOS_PRIZE_PAIJAN.kmz"
)


@st.cache_data(show_spinner="Descargando KMZ desde GitHub…")
def download_kmz_from_github(api_url: str, token: str) -> bytes:
    """
    Descarga el KMZ desde GitHub vía Contents API.
    Usa Accept: application/vnd.github.v3.raw para obtener
    los bytes crudos del archivo directamente.
    Cacheado: solo descarga una vez por sesión.
    """
    import urllib.request, urllib.error

    headers = {"Accept": "application/vnd.github.v3.raw"}
    if token:
        headers["Authorization"] = f"token {token}"

    try:
        req  = urllib.request.Request(api_url, headers=headers)
        resp = urllib.request.urlopen(req, timeout=30)
        return resp.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"Error descargando KMZ desde GitHub ({e.code}): {e.reason}. "
            f"Verifica que GITHUB_TOKEN esté en .streamlit/secrets.toml "
            f"y tenga permisos de lectura sobre el repo."
        )
    except Exception as ex:
        raise RuntimeError(f"Error de red al descargar KMZ: {ex}")


df = load_trampas_anexadas()

try:
    kmz_bytes_raw = download_kmz_from_github(KMZ_API_URL, GITHUB_TOKEN_KMZ)
    kml_bytes     = extract_kml_from_kmz_bytes(kmz_bytes_raw)
    polygons      = load_polygons_from_kml(kml_bytes)

except Exception as e:
    st.error(f"Error leyendo KMZ: {e}")
    polygons = []

if polygons:
    union_polys    = unary_union([p["shapely_polygon"] for p in polygons])
    prepared_union = prep(union_polys)
else:
    union_polys    = None
    prepared_union = None


# ============================================================
# SIDEBAR — RESET
# ============================================================
st.sidebar.header("⚙️ Configuración")

if st.sidebar.button("🔄 Limpiar todos los filtros", use_container_width=True):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

st.sidebar.markdown("---")


# ============================================================
# FILTROS ENCADENADOS
# ============================================================
with st.sidebar.expander("🔍 Filtros de datos", expanded=True):

    anios_opts = sorted(df["anio"].dropna().unique().astype(int).tolist())
    sel_anio   = st.multiselect("Año", options=anios_opts, default=[])
    df_f = df[df["anio"].isin(sel_anio)].copy() if sel_anio else df.copy()

    semanas_opts = sorted(df_f["semana"].dropna().unique().astype(int).tolist())
    sel_semana   = st.multiselect("Semana", options=semanas_opts, default=[])
    df_f = df_f[df_f["semana"].isin(sel_semana)].copy() if sel_semana else df_f

    fundos_opts = sorted(df_f["fundo"].dropna().unique().tolist())
    sel_fundo   = st.multiselect("Fundo", options=fundos_opts, default=[])
    df_f = df_f[df_f["fundo"].isin(sel_fundo)].copy() if sel_fundo else df_f

    mods_opts = sorted(df_f["modulo"].dropna().unique().tolist())
    sel_mod   = st.multiselect("Módulo", options=mods_opts, default=[])
    df_f = df_f[df_f["modulo"].isin(sel_mod)].copy() if sel_mod else df_f

    lotes_opts = sorted(df_f["lote"].dropna().unique().tolist()) if "lote" in df_f.columns else []
    sel_lote   = st.multiselect("Lote", options=lotes_opts, default=[])
    df_f = df_f[df_f["lote"].isin(sel_lote)].copy() if sel_lote else df_f

    turnos_opts = sorted(df_f["turno"].dropna().unique().tolist())
    sel_turno   = st.multiselect("Turno", options=turnos_opts, default=[])
    df_f = df_f[df_f["turno"].isin(sel_turno)].copy() if sel_turno else df_f

    trampas_opts = sorted(df_f["trampa"].dropna().unique().tolist())
    sel_trampa   = st.multiselect("Trampa", options=trampas_opts, default=[])
    df_f = df_f[df_f["trampa"].isin(sel_trampa)].copy() if sel_trampa else df_f

    if "fecha" in df_f.columns and not df_f.empty:
        min_f = df_f["fecha"].min()
        max_f = df_f["fecha"].max()
        sel_fecha = st.date_input(
            "Rango de fechas", value=(min_f, max_f),
            min_value=min_f, max_value=max_f
        )
        f_ini, f_fin = (
            (sel_fecha[0], sel_fecha[1])
            if isinstance(sel_fecha, tuple)
            else (sel_fecha, sel_fecha)
        )
        dff = df_f[(df_f["fecha"] >= f_ini) & (df_f["fecha"] <= f_fin)].copy()
    else:
        dff = df_f.copy()

    st.markdown("**Filtro por semaforización:**")
    sel_sem_verde    = st.checkbox("🟢 0 capturas",   value=True, key="sem_verde")
    sel_sem_amarillo = st.checkbox("🟡 1 captura",    value=True, key="sem_amarillo")
    sel_sem_naranja  = st.checkbox("🟠 2 capturas",   value=True, key="sem_naranja")
    sel_sem_rojo_f   = st.checkbox("🔴 > 2 capturas", value=True, key="sem_rojo")

    cats_permitidas = set()
    if sel_sem_verde:    cats_permitidas.add(0)
    if sel_sem_amarillo: cats_permitidas.add(1)
    if sel_sem_naranja:  cats_permitidas.add(2)
    if sel_sem_rojo_f:   cats_permitidas.add(3)


# FIX 2 aplicado: filtro KMZ vectorizado
if not dff.empty and union_polys is not None:
    dff = filter_within_union(dff, union_polys)

if not dff.empty:
    dff = (
        dff
        .groupby(["lat", "lon", "fundo", "modulo", "turno", "trampa"], as_index=False)
        .agg({"capturas": "sum"})
    )

if not dff.empty:
    dff["_cat"] = dff["capturas"].apply(get_semaforo_category)
    dff = dff[dff["_cat"].isin(cats_permitidas)].drop(columns=["_cat"])

st.sidebar.markdown("---")


# ============================================================
# MODO DE VISUALIZACIÓN
# ============================================================
modo_color = st.sidebar.radio(
    "🎨 Modo de visualización",
    options=["Normal", "Espectral", "Curvas de Nivel"],
    index=0
)

num_niveles = grosor_lineas = opacidad_relleno = None
mostrar_etiquetas = False
if modo_color == "Curvas de Nivel":
    with st.sidebar.expander("⚙️ Opciones curvas de nivel", expanded=True):
        num_niveles       = st.slider("Número de niveles",  2, 20, 5)
        grosor_lineas     = st.slider("Grosor de líneas",   0,  5, 1)
        mostrar_etiquetas = st.checkbox("Mostrar etiquetas de valor", value=False)
        opacidad_relleno  = st.slider("Opacidad relleno",   0, 100, 33)

buffer_val = st.sidebar.slider("📏 Buffer trampas (grados)", 0.001, 0.05, 0.010, step=0.001)
st.sidebar.markdown("---")


# ============================================================
# VECTORES DE PROPAGACIÓN
# ============================================================
with st.sidebar.expander("🧭 Vectores de Propagación", expanded=False):
    mostrar_vectores = st.checkbox(
        "Mostrar saetas de propagación", value=False, key="show_vectors"
    )
    n_arrows      = 15
    escala_flecha = 6
    min_mag       = 0.05
    color_flechas = "#1a1aff"
    head_size     = 6

    if mostrar_vectores:
        n_arrows      = st.slider("Densidad de saetas",              5,  30, 15, key="n_arrows")
        escala_flecha = st.slider("Longitud de saetas (×10⁻⁴ °)",   1,  20,  6, key="arrow_scale")
        head_size     = st.slider("Tamaño de punta (×10⁻⁵ °)",      1,  20,  6, key="head_size")
        min_mag       = st.slider("Magnitud mínima (filtro ruido)",  1,  30,  5, key="min_mag") / 100.0
        color_flechas = st.color_picker("Color saetas", value="#1a1aff", key="arrow_color")
        st.caption("🧭 Las saetas apuntan hacia donde **aumenta** la densidad de captura.")

st.sidebar.markdown("---")
st.sidebar.info(f"📊 **Registros encontrados:** {len(dff)}")

filtros_activos = []
if sel_anio:   filtros_activos.append(f"**Año:** {', '.join(map(str, sel_anio))}")
if sel_semana: filtros_activos.append(f"**Semana:** {', '.join(map(str, sel_semana))}")
if sel_fundo:  filtros_activos.append(f"**Fundo:** {', '.join(sel_fundo)}")
if sel_mod:    filtros_activos.append(f"**Módulo:** {', '.join(sel_mod)}")
if sel_lote:   filtros_activos.append(f"**Lote:** {', '.join(sel_lote)}")
if sel_turno:  filtros_activos.append(f"**Turno:** {', '.join(sel_turno)}")
if sel_trampa: filtros_activos.append(f"**Trampa:** {', '.join(sel_trampa)}")
if filtros_activos:
    st.sidebar.success("✅ Filtros activos")
    with st.sidebar.expander("📋 Ver filtros activos"):
        for fa in filtros_activos:
            st.markdown(fa)


# ============================================================
# DATOS ACTIVOS
# ============================================================
dff_activa = dff


# ============================================================
# LEYENDA INTERACTIVA — Alertas Rojas por Fundo/Turno
# ============================================================
if not dff_activa.empty and "fundo" in dff_activa.columns:
    dff_rojos = dff_activa[dff_activa["capturas"] > 2].copy()
    if not dff_rojos.empty:
        resumen_rojo = (
            dff_rojos
            .groupby(["fundo", "turno"])["capturas"]
            .sum()
            .reset_index()
        )
        total_fundo_df = (
            resumen_rojo.groupby("fundo")["capturas"].sum()
            .reset_index().rename(columns={"capturas": "total_fundo"})
            .sort_values("total_fundo", ascending=False)
        )
        resumen_rojo = (
            resumen_rojo
            .merge(total_fundo_df, on="fundo")
            .sort_values(["total_fundo", "fundo", "capturas"],
                         ascending=[False, True, False])
            .reset_index(drop=True)
        )
        resumen_rojo["_key"] = resumen_rojo["fundo"] + "||" + resumen_rojo["turno"]
        turnos_rojos_keys    = resumen_rojo["_key"].tolist()
    else:
        resumen_rojo      = pd.DataFrame(columns=["fundo","turno","capturas","total_fundo","_key"])
        turnos_rojos_keys = []
else:
    resumen_rojo      = pd.DataFrame(columns=["fundo","turno","capturas","total_fundo","_key"])
    turnos_rojos_keys = []

for k in turnos_rojos_keys:
    if f"tr_{k}" not in st.session_state:
        st.session_state[f"tr_{k}"] = True

if turnos_rojos_keys:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔴 Capturas > 2 por Fundo/Turno")
    st.sidebar.markdown(
        "<small style='color:gray;'>Agrupado Fundo → Turno (mayor → menor).</small>",
        unsafe_allow_html=True
    )

    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        if st.button("✅ Todos", use_container_width=True, key="_btn_tr_all"):
            for k in turnos_rojos_keys:
                st.session_state[f"tr_{k}"] = True
            st.rerun()
    with col_b:
        if st.button("❌ Ninguno", use_container_width=True, key="_btn_tr_none"):
            for k in turnos_rojos_keys:
                st.session_state[f"tr_{k}"] = False
            st.rerun()

    for fundo_nombre, grupo_df in resumen_rojo.groupby("fundo", sort=False):
        total_f   = int(grupo_df["capturas"].sum())
        max_cap_f = int(grupo_df["capturas"].max()) if not grupo_df.empty else 1

        st.sidebar.markdown(
            f"<div style='font-size:11px;font-weight:700;color:#900;"
            f"background:#fff0f0;border-radius:5px;padding:4px 8px;"
            f"margin:8px 0 3px 0;border-left:4px solid #FF0000;"
            f"display:flex;justify-content:space-between;align-items:center;'>"
            f"<span>📍 {escape(fundo_nombre)}</span>"
            f"<span style='background:#FF0000;color:#fff;border-radius:9px;"
            f"padding:0 7px;font-size:10px;'>{total_f} 🪰 total</span></div>",
            unsafe_allow_html=True
        )

        for _, row in grupo_df.iterrows():
            turno_name = row["turno"]
            cap_tot    = int(row["capturas"])
            key_r      = row["_key"]
            pct        = (cap_tot / max_cap_f * 100) if max_cap_f > 0 else 0

            col_chk, col_info = st.sidebar.columns([1, 5])
            with col_chk:
            
                st.checkbox(
                    "Seleccionar",
                    key=f"tr_{key_r}",
                    label_visibility="collapsed"
                )
            with col_info:
                visible = st.session_state.get(f"tr_{key_r}", True)
                tach    = "text-decoration:line-through;color:#bbb;" if not visible else ""
                opac    = "opacity:0.4;" if not visible else ""
                st.markdown(
                    f"""
                    <div style="display:flex;align-items:center;gap:5px;
                                padding:2px 0;font-size:12px;{opac}">
                        <span style="display:inline-block;width:13px;height:13px;
                                     background:#FF0000;border:1.5px solid #900;
                                     border-radius:3px;flex-shrink:0;"></span>
                        <span style="flex:1;overflow:hidden;text-overflow:ellipsis;
                                     white-space:nowrap;font-weight:600;{tach}"
                              title="{escape(turno_name)}">{escape(turno_name)}</span>
                        <span style="background:#ffe5e5;border:1px solid #f88;
                                     border-radius:9px;padding:1px 6px;
                                     font-weight:700;font-size:11px;
                                     color:#c00;white-space:nowrap;">{cap_tot} 🪰</span>
                    </div>
                    <div style="height:4px;border-radius:3px;margin:2px 0 4px 18px;
                                background:linear-gradient(to right,
                                #FF0000 {pct:.1f}%,#f0d0d0 {pct:.1f}%);
                                border:1px solid #f8b;{opac}"></div>
                    """,
                    unsafe_allow_html=True
                )

    vis_rojo_total = sum(
        int(r["capturas"]) for _, r in resumen_rojo.iterrows()
        if st.session_state.get(f"tr_{r['_key']}", True)
    )
    st.sidebar.markdown(
        f"<div style='text-align:right;font-size:11px;color:#c00;margin-top:6px;'>"
        f"<b>🔴 Total visible:</b> {vis_rojo_total} capturas</div>",
        unsafe_allow_html=True
    )
else:
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<small style='color:#888;'>ℹ️ No hay capturas > 2 en los datos actuales.</small>",
        unsafe_allow_html=True
    )

claves_visibles = {
    k for k in turnos_rojos_keys
    if st.session_state.get(f"tr_{k}", True)
}


def apply_visibility(df_in: pd.DataFrame) -> pd.DataFrame:
    if df_in.empty:
        return df_in
    def _vis(row):
        if float(row["capturas"]) > 2:
            return f"{row['fundo']}||{row['turno']}" in claves_visibles
        return True
    return df_in[df_in.apply(_vis, axis=1)].copy()

dff_map = apply_visibility(dff_activa)


# ============================================================
# MAPA BASE
# ============================================================
ref_df     = dff_map if not dff_map.empty else (dff if not dff.empty else df)
center_lat = float(ref_df["lat"].mean())
center_lon = float(ref_df["lon"].mean())

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=15,
    control_scale=True,
    prefer_canvas=True,
    # CartoDB Voyager: permite CORS → tiles capturables en PNG exportado
    tiles="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
    attr='&copy; OpenStreetMap &copy; CARTO',
)

m.get_root().html.add_child(folium.Element("""
<style>
@keyframes fly {
    0%   { transform: translate(0px,  0px) rotate(0deg)   scale(1);    }
    15%  { transform: translate(3px, -4px) rotate(15deg)  scale(1.1);  }
    30%  { transform: translate(-2px,-7px) rotate(-10deg) scale(0.95); }
    45%  { transform: translate(5px, -3px) rotate(20deg)  scale(1.05); }
    60%  { transform: translate(-3px,-5px) rotate(-15deg) scale(1);    }
    75%  { transform: translate(2px, -2px) rotate(10deg)  scale(1.08); }
    100% { transform: translate(0px,  0px) rotate(0deg)   scale(1);    }
}
</style>
"""))

if polygons:
    fg_poly  = FeatureGroup(name="Módulos (KMZ - bordes)", show=True)
    cmap_mod = matplotlib.cm.get_cmap("tab20", max(len(polygons), 1))
    for i, p in enumerate(polygons):
        folium.Polygon(
            locations=p["folium_coords"],
            tooltip=p["name"],
            color=matplotlib.colors.rgb2hex(cmap_mod(i)[:3]),
            weight=3, fill=False
        ).add_to(fg_poly)
    fg_poly.add_to(m)


# ============================================================
# FIX 4: HEATMAP — GRID_RES=200 + mask_poly vectorizado
#
# GRID_RES=200 → 40 000 pts (vs 360 000 con 600).
# contains_xy de shapely>=2 vectoriza la máscara sin loop Python.
# ============================================================
GRID_RES = 600

grid_z_for_vectors = None
grid_x_gv = grid_y_gv = mask_poly_gv = None

if (
    not dff_map.empty
    and union_polys is not None
    and len(dff_map) >= 3
    and dff_map["capturas"].sum() > 0
):
    x = dff_map["lon"].values
    y = dff_map["lat"].values
    z = dff_map["capturas"].values.astype(float)

    max_capturas = max(z.max(), 1)
    eps          = 1e-6
    xmin, xmax   = x.min(), x.max()
    ymin, ymax   = y.min(), y.max()
    if np.isclose(xmin, xmax): xmin -= eps; xmax += eps
    if np.isclose(ymin, ymax): ymin -= eps; ymax += eps

    grid_x, grid_y = np.mgrid[
        xmin:xmax:complex(0, GRID_RES),
        ymin:ymax:complex(0, GRID_RES)
    ]

    try:
        if modo_color == "Normal":
            grid_z = griddata((x, y), z, (grid_x, grid_y), method="nearest")
        else:
            grid_z = griddata((x, y), z, (grid_x, grid_y), method="linear")
            mask_nan = np.isnan(grid_z)
            if mask_nan.any():
                grid_z[mask_nan] = griddata(
                    (x, y), z, (grid_x, grid_y), method="nearest"
                )[mask_nan]
    except Exception:
        grid_z = None

    if grid_z is not None:
        # Máscara de polígono vectorizada (FIX 4)
        try:
            from shapely import contains_xy as _contains_xy
            mask_poly = _contains_xy(
                union_polys,
                grid_x.ravel(),
                grid_y.ravel()
            ).reshape(grid_z.shape)
        except ImportError:
            # shapely < 2.0 — fallback con prepared geometry
            _prep = prep(union_polys)
            pts_flat = np.vstack((grid_x.ravel(), grid_y.ravel())).T
            mask_poly = np.array(
                [_prep.contains(Point(lx, ly)) for lx, ly in pts_flat]
            ).reshape(grid_z.shape)

        grid_z_masked = grid_z.copy()
        grid_z_masked[~mask_poly] = np.nan

        gz_smooth_global             = gaussian_filter(
            np.where(np.isnan(grid_z_masked), 0, grid_z_masked), sigma=8
        )
        gz_smooth_global[~mask_poly] = np.nan
        grid_z_for_vectors           = gz_smooth_global
        grid_x_gv, grid_y_gv, mask_poly_gv = grid_x, grid_y, mask_poly

        if not np.all(np.isnan(grid_z_masked)):

            if modo_color == "Curvas de Nivel":
                fig, ax = plt.subplots(figsize=(GRID_RES/100, GRID_RES/100), dpi=100)
                ax.set_axis_off()
                fig.patch.set_alpha(0)

                gz_smooth = gaussian_filter(
                    np.where(np.isnan(grid_z_masked), 0, grid_z_masked), sigma=8
                )
                gz_smooth[~mask_poly] = np.nan
                max_level = max(4.0, float(max_capturas) + 1)

                try:
                    ax.contourf(
                        grid_x.T, grid_y.T, gz_smooth.T,
                        levels=[0, 1, 2, 3, max_level],
                        colors=["#00FF00","#FFFF00","#FFA500","#FF0000"],
                        alpha=opacidad_relleno/100, extend="max"
                    )
                    if grosor_lineas > 0:
                        cl = ax.contour(
                            grid_x.T, grid_y.T, gz_smooth.T,
                            levels=np.linspace(
                                0, float(max_capturas), num_niveles+1
                            ).tolist(),
                            colors="black",
                            linewidths=float(grosor_lineas),
                            alpha=0.75
                        )
                        if mostrar_etiquetas:
                            ax.clabel(cl, inline=True, fontsize=8,
                                      fmt="%.0f", colors="black")
                except Exception:
                    pass

                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                plt.tight_layout(pad=0)
                buf = io.BytesIO()
                fig.savefig(buf, format="PNG", bbox_inches="tight",
                            pad_inches=0, transparent=True, dpi=150)
                plt.close(fig)
                buf.seek(0)
                b64 = base64.b64encode(buf.read()).decode()
                folium.raster_layers.ImageOverlay(
                    image="data:image/png;base64," + b64,
                    bounds=[[ymin, xmin], [ymax, xmax]],
                    opacity=1.0, name="Curvas de Nivel"
                ).add_to(m)

            else:
                rgba  = np.zeros(grid_z_masked.shape + (4,), dtype=np.uint8)
                valid = ~np.isnan(grid_z_masked)
                for i in range(grid_z_masked.shape[0]):
                    for j in range(grid_z_masked.shape[1]):
                        if valid[i, j]:
                            ch  = (get_color_espectral(grid_z_masked[i, j])
                                   if modo_color == "Espectral"
                                   else get_color_normal(grid_z_masked[i, j]))
                            rgb = mcolors.to_rgb(ch)
                            rgba[i, j, :3] = (np.array(rgb) * 255).astype(np.uint8)
                            rgba[i, j,  3] = 180
                rgba[~mask_poly] = [0, 0, 0, 0]

                pil_img = PILImage.fromarray(
                    np.flipud(rgba.transpose(1, 0, 2)), mode="RGBA"
                )
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                pil_img.save(tmp.name, format="PNG")
                tmp.close()
                with open(tmp.name, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                os.unlink(tmp.name)

                folium.raster_layers.ImageOverlay(
                    image="data:image/png;base64," + b64,
                    bounds=[[ymin, xmin], [ymax, xmax]],
                    opacity=1.0, name=f"Heatmap ({modo_color})"
                ).add_to(m)

        else:
            st.info("No hay capturas dentro del KMZ seleccionado.")


# ============================================================
# SAETAS DE PROPAGACIÓN
# ============================================================
if mostrar_vectores and grid_z_for_vectors is not None and grid_x_gv is not None:
    fg_vec  = FeatureGroup(name="🧭 Saetas de Propagación", show=True)
    escala  = escala_flecha / 10000.0
    hs_deg  = head_size / 100000.0

    vectors = compute_gradient_vectors(
        grid_x_gv, grid_y_gv, grid_z_for_vectors, mask_poly_gv,
        n_arrows=n_arrows, min_magnitude=min_mag
    )

    if vectors:
        mags    = np.array([v["magnitude"] for v in vectors], dtype=float)
        mags_ok = mags[np.isfinite(mags)]
        mag_max = float(mags_ok.max()) if len(mags_ok) > 0 and mags_ok.max() > 0 else 1.0

        for v in vectors:
            raw_mag = v["magnitude"]
            if not math.isfinite(raw_mag) or not math.isfinite(v["dlat"]) \
                    or not math.isfinite(v["dlon"]):
                continue

            mag_norm  = float(raw_mag) / mag_max
            mag_norm  = max(0.0, min(1.0, mag_norm))
            arrow_len = escala * (0.3 + 0.7 * mag_norm)
            lat0, lon0 = v["lat"], v["lon"]
            lat1 = lat0 + v["dlat"] * arrow_len
            lon1 = lon0 + v["dlon"] * arrow_len
            opacity   = 0.45 + 0.55 * mag_norm
            weight    = 1 + int(mag_norm * 3)
            tip_txt   = (
                f"Densidad: {v['density']:.2f} | "
                f"Gradiente: {v['magnitude']:.3f}"
            )

            draw_arrow(
                fg=fg_vec,
                lat0=lat0, lon0=lon0,
                lat1=lat1, lon1=lon1,
                color=color_flechas,
                weight=weight,
                opacity=opacity,
                head_size_deg=hs_deg * (0.6 + 0.4 * mag_norm),
                tooltip=tip_txt
            )

    fg_vec.add_to(m)


# ============================================================
# MARCADORES DE TRAMPAS
# ============================================================
if not dff_map.empty:
    fg_pts = FeatureGroup(name="Trampas", show=True)
    for _, r in dff_map.iterrows():
        cap       = float(r.get("capturas", 0))
        color     = (get_color_espectral(cap)
                     if modo_color == "Espectral"
                     else get_color_normal(cap))
        turno_txt = escape(str(r.get("turno", "")))
        tip = (
            f"<b>Fundo:</b> {escape(r.get('fundo',''))}<br>"
            f"<b>Módulo:</b> {escape(r.get('modulo',''))}<br>"
            f"<b>Turno:</b> {turno_txt}<br>"
            f"<b>Trampa:</b> {escape(r.get('trampa',''))}<br>"
            f"<b>Capturas:</b> {int(cap)}"
        )
        folium.CircleMarker(
            location=(r["lat"], r["lon"]), radius=8,
            color="black", weight=1, fill=True,
            fill_color=color, fill_opacity=0.9, tooltip=tip
        ).add_to(fg_pts)

        if cap > 2:
            label_html = (
                f'<div style="font-size:7pt;font-weight:bold;color:#222;'
                f'text-shadow:1px 1px 1px #fff,-1px -1px 1px #fff;'
                f'white-space:nowrap;line-height:1.3;'
                f'display:flex;align-items:center;gap:2px;">'
                f'{turno_txt} {int(cap)}'
                f'<span style="display:inline-block;'
                f'animation:fly 1.4s ease-in-out infinite;font-size:9pt;">🪰</span>'
                f'</div>'
            )
        else:
            label_html = (
                f'<div style="font-size:7pt;font-weight:bold;color:#222;'
                f'text-shadow:1px 1px 1px #fff,-1px -1px 1px #fff;'
                f'white-space:nowrap;line-height:1.1;">'
                f'{turno_txt}</div>'
            )

        folium.Marker(
            location=(r["lat"], r["lon"]),
            icon=folium.DivIcon(
                html=label_html, icon_size=(0, 0), icon_anchor=(0, -12)
            )
        ).add_to(fg_pts)

    fg_pts.add_to(m)


# ============================================================
# LEYENDA FLOTANTE — Alertas Fundo/Turno
# ============================================================
if not resumen_rojo.empty:
    filas_mapa = ""
    for fundo_m, grupo_m in resumen_rojo.groupby("fundo", sort=False):
        total_m   = int(grupo_m["capturas"].sum())
        max_cap_m = int(grupo_m["capturas"].max()) if not grupo_m.empty else 1

        filas_mapa += (
            f'<div style="font-size:10px;font-weight:700;color:#900;'
            f'border-left:3px solid #FF0000;padding:2px 6px;'
            f'margin:6px 0 2px 0;background:#fff5f5;border-radius:3px;'
            f'display:flex;justify-content:space-between;align-items:center;">'
            f'<span>📍 {escape(fundo_m)}</span>'
            f'<span style="background:#FF0000;color:#fff;border-radius:8px;'
            f'padding:0 6px;font-size:9px;">{total_m} 🪰</span></div>'
        )
        for _, row in grupo_m.iterrows():
            turno_name = row["turno"]
            cap_tot    = int(row["capturas"])
            key_r      = row["_key"]
            visible    = st.session_state.get(f"tr_{key_r}", True)
            opac_sty   = "" if visible else "opacity:0.3;"
            tach_sty   = "" if visible else "text-decoration:line-through;color:#bbb;"
            pct_m      = (cap_tot / max_cap_m * 100) if max_cap_m > 0 else 0

            filas_mapa += f"""
            <div style="display:flex;align-items:center;gap:4px;
                        margin-bottom:2px;padding-left:8px;{opac_sty}">
                <span style="display:inline-block;width:10px;height:10px;
                             background:#FF0000;border:1px solid #900;
                             border-radius:2px;flex-shrink:0;"></span>
                <span style="flex:1;font-size:10px;font-weight:600;
                             overflow:hidden;text-overflow:ellipsis;
                             white-space:nowrap;max-width:90px;{tach_sty}"
                      title="{escape(turno_name)}">{escape(turno_name)}</span>
                <span style="font-size:9px;font-weight:700;background:#ffe5e5;
                             border:1px solid #f99;border-radius:8px;padding:0 4px;
                             color:#c00;white-space:nowrap;">{cap_tot} 🪰</span>
            </div>
            <div style="height:3px;border-radius:2px;margin:1px 0 3px 18px;
                        background:linear-gradient(to right,
                        #FF0000 {pct_m:.1f}%,#f0d0d0 {pct_m:.1f}%);
                        border:1px solid #f8b;{opac_sty}"></div>
            """

    vis_tot_mapa = sum(
        int(r["capturas"]) for _, r in resumen_rojo.iterrows()
        if st.session_state.get(f"tr_{r['_key']}", True)
    )

    m.get_root().html.add_child(folium.Element(f"""
    <div style="position:fixed;top:175px;right:10px;z-index:999;
        background:rgba(255,255,255,0.96);border:2px solid #c33;
        border-radius:8px;padding:8px 12px;
        box-shadow:2px 2px 8px rgba(0,0,0,0.25);
        font-family:Arial,sans-serif;min-width:200px;max-width:240px;
        max-height:380px;overflow-y:auto;">
        <div style="font-size:12px;font-weight:bold;color:#c00;margin-bottom:2px;">
            🔴 Alertas Fundo / Turno
        </div>
        <div style="font-size:9px;color:#888;margin-bottom:5px;">
            Capturas &gt; 2 · Fundo → Turno (mayor → menor)
        </div>
        <hr style="margin:4px 0;border:none;border-top:1px solid #f8d;">
        {filas_mapa}
        <hr style="margin:4px 0;border:none;border-top:1px solid #f8d;">
        <div style="font-size:10px;color:#c00;text-align:right;font-weight:700;">
            Total visible: {vis_tot_mapa} 🪰
        </div>
    </div>
    """))


# ============================================================
# LEYENDA SAETAS
# ============================================================
if mostrar_vectores:
    m.get_root().html.add_child(folium.Element(f"""
    <div style="position:fixed;top:0px;left:10px;z-index:999;
        background:rgba(255,255,255,0.93);border:2px solid {color_flechas};
        border-radius:19px;padding:9px 14px;
        box-shadow:2px 2px 8px rgba(0,0,0,0.25);
        font-family:Arial,sans-serif;min-width:190px;">
        <div style="font-size:12px;font-weight:bold;color:#333;margin-bottom:4px;">
            🧭 Saetas de Propagación
        </div>
        <div style="font-size:10px;color:#555;line-height:1.7;">
            ➤ <b>Saeta corta, delgada</b> → flujo bajo<br>
            ➤➤ <b>Saeta larga, gruesa</b> → expansión activa<br>
            <span style="color:{color_flechas};font-weight:700;">▶</span>
            Dirección del gradiente de densidad
        </div>
        <div style="font-size:9px;color:#888;margin-top:4px;">
            Apuntan hacia donde <b>aumenta</b> la concentración.
        </div>
    </div>
    """))


# ============================================================
# LEYENDA SEMAFORIZACIÓN
# ============================================================
m.get_root().html.add_child(folium.Element(f"""
<div style="position:fixed;bottom:160px;right:10px;z-index:1000;
    background:rgba(255,255,255,0.93);border:2px solid #555;
    border-radius:8px;padding:10px 16px;font-size:13px;
    box-shadow:2px 2px 8px rgba(0,0,0,0.3);
    font-family:Arial,sans-serif;min-width:200px;">
  <b>Umbral de Capturas</b>
  <span style="font-size:10px;color:#666;"> ({modo_color})</span><br><br>
  <span style="display:inline-block;width:18px;height:18px;background:#00FF00;
        border:1px solid #333;vertical-align:middle;margin-right:8px;
        border-radius:3px;"></span>0 capturas<br><br>
  <span style="display:inline-block;width:18px;height:18px;background:#FFFF00;
        border:1px solid #333;vertical-align:middle;margin-right:8px;
        border-radius:3px;"></span>1 captura<br><br>
  <span style="display:inline-block;width:18px;height:18px;background:#FFA500;
        border:1px solid #333;vertical-align:middle;margin-right:8px;
        border-radius:3px;"></span>2 capturas<br><br>
  <span style="display:inline-block;width:18px;height:18px;background:#FF0000;
        border:1px solid #333;vertical-align:middle;margin-right:8px;
        border-radius:3px;"></span>&gt; 2 capturas
</div>
"""))


# ============================================================
# CONTROL DE CAPAS
# ============================================================
folium.LayerControl(collapsed=False).add_to(m)


# ============================================================
# BOTÓN PNG FLOTANTE — inyectado dentro del iframe del mapa
# Usa html2canvas (CDN) para capturar el canvas de Leaflet
# y descargarlo como PNG directamente desde el browser.
# ============================================================
# Construir etiqueta de filtros para el nombre del archivo
_partes_nombre = []
if sel_anio:   _partes_nombre.append("A" + "-".join(map(str, sel_anio)))
if sel_semana: _partes_nombre.append("S" + "-".join(map(str, sel_semana)))
if sel_fundo:  _partes_nombre.append("F" + "-".join(sel_fundo))
if sel_mod:    _partes_nombre.append("M" + "-".join(sel_mod))
if sel_lote:   _partes_nombre.append("L" + "-".join(sel_lote))
if sel_turno:  _partes_nombre.append("T" + "-".join(sel_turno))
_nombre_png = ("MosaicaFruta_" + "_".join(_partes_nombre) if _partes_nombre
               else "MosaicaFruta_TodosLosDatos")
# Limpiar caracteres no válidos en nombre de archivo
import re as _re_fn
_nombre_png = _re_fn.sub(r'[^A-Za-z0-9_\-]', '', _nombre_png)[:120]

m.get_root().html.add_child(folium.Element(f"""
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>


<div id="export-spinner" style="display:none;position:fixed;bottom:20px;
    left:50%;transform:translateX(-50%);z-index:2000;
    background:rgba(13,35,64,0.92);color:#fff;border-radius:8px;
    padding:10px 18px;font-size:12px;font-family:Arial,sans-serif;">
    ⏳ Capturando mapa…
</div>

<script>
function exportarPNG() {{
    var btn     = document.getElementById('btn-png');
    var spinner = document.getElementById('export-spinner');
    btn.style.display     = 'none';
    spinner.style.display = 'block';

    setTimeout(function() {{
        html2canvas(document.body, {{
            useCORS: true,
            allowTaint: false,
            backgroundColor: '#ffffff',
            scale: 2,
            ignoreElements: function(el) {{
                return el.id === 'btn-png' || el.id === 'export-spinner';
            }}
        }}).then(function(canvas) {{
            var link      = document.createElement('a');
            link.download = '{_nombre_png}.png';
            link.href     = canvas.toDataURL('image/png');
            link.click();
            btn.style.display     = 'block';
            spinner.style.display = 'none';
        }}).catch(function(err) {{
            alert('Error: ' + err);
            btn.style.display     = 'block';
            spinner.style.display = 'none';
        }});
    }}, 800);
}}
</script>
"""))

# ============================================================
# EXPORTAR → GITHUB PAGES
# Sube mapa_mosca.html a la rama gh-pages del repo via API.
# La URL es siempre la misma → Power BI / browser la usan fija.
# ─────────────────────────────────────────────────────────────
# Setup inicial (solo 1 vez):
#   1. Crea el token en github.com → Settings → Developer settings
#      → Personal access tokens → Fine-grained
#      Permisos necesarios: Contents R/W, Pages R/W, Workflows R/W
#   2. Copia secrets.toml a .streamlit/secrets.toml
#   3. Corre: python setup_ghpages.py   (una sola vez)
# ============================================================
_html_mapa   = m._repr_html_()
_nombre_html = _nombre_png + ".html"

# Configuracion desde .streamlit/secrets.toml
GITHUB_TOKEN  = st.secrets.get("GITHUB_TOKEN",  "")
GITHUB_OWNER  = st.secrets.get("GITHUB_OWNER",  "controloperacionalprize-boss")
GITHUB_REPO   = st.secrets.get("GITHUB_REPO",   "mapa_html")
GITHUB_BRANCH = st.secrets.get("GITHUB_BRANCH", "main")
GITHUB_FILE   = "mapa_mosca.html"   # nombre FIJO — la URL nunca cambia

# URL publica permanente (siempre la misma sin importar filtros)
GITHUB_PAGES_URL = (
    f"https://{GITHUB_OWNER}.github.io/{GITHUB_REPO}/{GITHUB_FILE}"
)

def _push_file_github(api_url: str, html_content: str, branch: str, 
                       mensaje: str, headers: dict) -> tuple:
    """Sube un archivo a GitHub. Devuelve (True, commit_url) o (False, error)."""
    import base64, urllib.request, urllib.error, json

    sha = None
    try:
        req  = urllib.request.Request(api_url + f"?ref={branch}", headers=headers)
        resp = urllib.request.urlopen(req, timeout=10)
        sha  = json.loads(resp.read())["sha"]
    except urllib.error.HTTPError as e:
        if e.code != 404:
            return False, f"Error leyendo archivo ({e.code}): {e.read().decode()[:200]}"
    except Exception as ex:
        return False, f"Error de red: {ex}"

    payload = {
        "message": mensaje,
        "content": base64.b64encode(html_content.encode("utf-8")).decode(),
        "branch":  branch,
    }
    if sha:
        payload["sha"] = sha

    try:
        data = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(api_url, data=data, headers=headers, method="PUT")
        resp = urllib.request.urlopen(req, timeout=60)
        result = json.loads(resp.read())
        return True, result.get("commit", {}).get("html_url", "")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return False, f"GitHub API error {e.code}: {body[:400]}"
    except Exception as ex:
        return False, f"Error de red al subir: {ex}"
    
def _subir_a_github(html_content: str):
    import datetime

    if not GITHUB_TOKEN:
        return False, "No se encontró GITHUB_TOKEN en .streamlit/secrets.toml."

    headers = {
        "Authorization":        f"Bearer {GITHUB_TOKEN}",
        "Accept":               "application/vnd.github+json",
        "Content-Type":         "application/json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    base_repo = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents"
    ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── Nombre del histórico con semana/año del filtro ──────────
    if sel_semana and sel_anio:
        anio_str   = "-".join(map(str, sorted(sel_anio)))
        semana_str = "-".join(map(str, sorted(sel_semana)))
        sufijo     = f"A{anio_str}_S{semana_str}"
    elif sel_anio:
        sufijo = "A" + "-".join(map(str, sorted(sel_anio)))
    elif sel_semana:
        sufijo = "S" + "-".join(map(str, sorted(sel_semana)))
    else:
        sufijo = "SinFiltro"

    ts_file   = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    nombre_h  = f"historico/mapa_{sufijo}.html"
    mensaje   = f"Mapa actualizado {ts} | {sufijo}"

    # ── 1. Archivo FIJO (URL permanente para Power BI) ───────────
    ok1, res1 = _push_file_github(
        api_url  = f"{base_repo}/{GITHUB_FILE}",
        html_content = html_content,
        branch   = GITHUB_BRANCH,
        mensaje  = mensaje,
        headers  = headers,
    )
    if not ok1:
        return False, f"Error subiendo archivo fijo: {res1}"

    # ── 2. Archivo HISTÓRICO (copia con fecha+semana) ────────────
    ok2, res2 = _push_file_github(
        api_url  = f"{base_repo}/{nombre_h}",
        html_content = html_content,
        branch   = GITHUB_BRANCH,
        mensaje  = mensaje,
        headers  = headers,
    )

    url_historico = (
        f"https://{GITHUB_OWNER}.github.io/{GITHUB_REPO}/{nombre_h}"
    )

    if ok2:
        return True, (res1, url_historico)
    else:
        # El fijo se subió OK pero el histórico falló — no es crítico
        return True, (res1, f"Histórico falló: {res2}")
# UI sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 🌐 Publicar")


# Boton publicar + descarga local
col_pub, col_png = st.sidebar.columns([3, 3])

with col_pub:
    if st.button("🚀 Publicar mapa", use_container_width=True, key="btn_gh_publish"):
        with st.spinner("Subiendo a GitHub Pages..."):
            ok, resultado = _subir_a_github(_html_mapa)
        if ok:
            st.sidebar.success("✅ Publicado.")
        else:
            st.sidebar.error(f"Error: {resultado}")

with col_png:

    if st.button("🖼️ PNG", use_container_width=True, key="btn_png"):

        with st.spinner("Capturando mapa..."):

            driver = None
            tmp_html = None

            try:

                from selenium import webdriver
                from selenium.webdriver.chrome.options import Options
                from selenium.webdriver.chrome.service import Service

                import platform
                import time
                import tempfile
                import os

                # HTML temporal
                tmp_html = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".html"
                )

                tmp_html.write(_html_mapa.encode("utf-8"))
                tmp_html.close()

                # Config Chrome
                opts = Options()

                opts.add_argument("--headless=new")
                opts.add_argument("--no-sandbox")
                opts.add_argument("--disable-dev-shm-usage")
                opts.add_argument("--disable-gpu")
                opts.add_argument("--window-size=1920,1080")

                # Windows local
                if platform.system() == "Windows":

                    from webdriver_manager.chrome import ChromeDriverManager

                    service = Service(
                        ChromeDriverManager().install()
                    )

                # Linux / Streamlit Cloud
                else:

                    opts.binary_location = "/usr/bin/chromium"

                    service = Service(
                        "/usr/bin/chromedriver"
                    )

                # Driver
                driver = webdriver.Chrome(
                    service=service,
                    options=opts
                )

                # Abrir HTML
                driver.get(f"file:///{tmp_html.name}")

                time.sleep(5)

                # Tamaño real
                total_width = driver.execute_script(
                    "return document.body.scrollWidth"
                )

                total_height = driver.execute_script(
                    "return document.body.scrollHeight"
                )

                driver.set_window_size(
                    total_width,
                    total_height
                )

                time.sleep(2)

                driver.execute_script(
                    "window.scrollTo(0, 0);"
                )

                time.sleep(1)

                # Screenshot
                png_bytes = driver.get_screenshot_as_png()

                # Descargar
                st.sidebar.download_button(
                    label="⬇️ Descargar PNG",
                    data=png_bytes,
                    file_name=f"{_nombre_png}.png",
                    mime="image/png",
                    key="btn_dl_png_real"
                )

                st.sidebar.success("✅ PNG listo")

            except Exception as e:

                st.sidebar.error(f"Error: {e}")

            finally:

                try:
                    if driver:
                        driver.quit()
                except:
                    pass

                try:
                    if tmp_html:
                        os.unlink(tmp_html.name)
                except:
                    pass

# ============================================================
# ST_FOLIUM
# ============================================================
st_folium(
    m,
    width=None,
    height=950,
    returned_objects=[],
    key="main_map"
)
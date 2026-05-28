# ============================================================
# MAPA EPIDEMIOLÓGICO AGRÍCOLA — Mosca de la Fruta
# v6 — Pintado por polígono KMZ (Módulo + Turno)
#   1. load_polygons_from_kml con @st.cache_data
#   2. KMZ leído como bytes (cacheable)
#   3. Filtro KMZ vectorizado (shapely>=2 contains_xy)
#   4. Pintado directo por polígono: cruza (mod_n, tur_n)
#      Excel ↔ KMZ con normalización robusta de texto sucio
#   5. st_folium con key="main_map"
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
# NORMALIZACIÓN MÓDULO / TURNO
# Cubre todos los formatos sucios encontrados en el Excel:
#   MOD 01 / MOD1 / M01 / MO 6 → número entero
#   M1-T6 / M06- T1 / T06 / '   M07-T12' / M4(A) → número entero
#   Descarta turno > 20 (son lotes mal ingresados en col Turno)
# ============================================================
import re as _re_norm

def _norm_mod(val) -> int | None:
    if val is None:
        return None
    try:
        if isinstance(val, float) and pd.isna(val):
            return None
    except Exception:
        pass
    m = _re_norm.search(r'(\d+)', str(val).upper().strip())
    return int(m.group(1)) if m else None

def _norm_tur(val) -> int | None:
    if val is None:
        return None
    try:
        if isinstance(val, float) and pd.isna(val):
            return None
    except Exception:
        pass
    s = str(val).upper().strip()
    # Primero buscar patrón T seguido de número (M1-T6, M06-T1, T06, etc.)
    m = _re_norm.search(r'T[\s\-]?(\d+)', s)
    if m:
        n = int(m.group(1))
        return n if n <= 20 else None
    # Fallback: solo número (ej. "6", "T01")
    m2 = _re_norm.search(r'^\s*T?(\d+)\s*$', s)
    if m2:
        n = int(m2.group(1))
        return n if n <= 20 else None
    return None

def _norm_lote(val) -> str | None:
    """
    Normaliza lote para cruzar Excel ↔ KMZ:
      1.0   → "1"
      10.0  → "10"
      115B  → "115B"   (Excel)
      115-b → "115B"   (KMZ: quitar guión, mayúsculas)
      64B   → "64B"
      7     → "7"
    """
    if val is None:
        return None
    import re as _re_l
    s = str(val).strip().upper()
    if not s or s in ('NAN', 'NONE', ''):
        return None
    # Float tipo "1.0" → "1"
    m = _re_l.match(r'^(\d+)\.0+$', s)
    if m:
        return m.group(1)
    # Quitar guiones internos: "115-B" → "115B"
    s = s.replace('-', '')
    return s


def _parse_desc_html(desc_html: str) -> dict:
    """
    Parsea la descripción HTML del placemark KMZ.
    Formato: pares <td>CLAVE</td><td>VALOR</td>
    Retorna dict con claves en minúsculas.
    """
    result = {}
    if not desc_html:
        return result
    pairs = _re_norm.findall(
        r'<td[^>]*>\s*([^<]+?)\s*</td>\s*<td[^>]*>\s*([^<]*?)\s*</td>',
        desc_html,
        _re_norm.IGNORECASE
    )
    for key, val in pairs:
        result[key.strip().lower()] = val.strip()
    return result


# ============================================================
# CARGA DE POLÍGONOS KMZ — con descripción HTML parseada
# Extrae Modulo, Turno, Lote, Area de cada placemark
# ============================================================
@st.cache_data(show_spinner="Leyendo KMZ…")
def load_polygons_with_desc(kml_bytes: bytes):
    import re as _re2

    kml_str = kml_bytes.decode("utf-8", errors="replace")
    if 'xmlns:xsi' not in kml_str:
        kml_str = _re2.sub(
            r'(<kml\b)',
            r'\1 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
            kml_str, count=1
        )
    kml_bytes_clean = kml_str.encode("utf-8")

    # Siempre usar lxml con recover=True — más robusto que pykml
    try:
        from lxml import etree
        root = etree.fromstring(
            kml_bytes_clean,
            parser=etree.XMLParser(recover=True)
        )
    except Exception as e:
        raise RuntimeError(f"No se pudo parsear el KML: {e}")

    ns = "http://www.opengis.net/kml/2.2"
    out = []

    def _tag(el):
        """Devuelve el tag sin namespace."""
        return el.tag.split("}")[-1] if "}" in el.tag else el.tag

    def _text(el, subtag):
        """Texto de un subelemento, o ''."""
        child = el.find(f"{{{ns}}}{subtag}")
        return (child.text or "").strip() if child is not None else ""

    def _extract_mod(name: str):
        """'AQ2 - MODULO 05' → 5,  'MODULO 3' → 3,  otro → None"""
        m = _re2.search(r'MODULO\s*0*(\d+)', name.upper())
        return int(m.group(1)) if m else None

    def _extract_fundo_aq(name: str):
        """'AQ2 - MODULO 05' → 'AQ2',  'AQ1 - MODULO 03' → 'AQ1',  otro → None"""
        m = _re2.search(r'\b(AQ\d+)\b', name.upper())
        return m.group(1) if m else None

    def _process_placemark(pm, inherited_mod_n, inherited_fundo_aq):
        pname   = _text(pm, "name")
        desc    = _text(pm, "description")
        attrs   = _parse_desc_html(desc)
        tur_n   = _norm_tur(attrs.get("turno"))

        # mod_n y fundo_aq: nombre propio > heredado del folder padre
        own_mod      = _extract_mod(pname)
        own_fundo_aq = _extract_fundo_aq(pname)
        mod_n      = own_mod      if own_mod      is not None else inherited_mod_n
        fundo_aq   = own_fundo_aq if own_fundo_aq is not None else inherited_fundo_aq

        for poly in pm.iter(f"{{{ns}}}Polygon"):
            outer = poly.find(
                f"{{{ns}}}outerBoundaryIs/{{{ns}}}LinearRing/{{{ns}}}coordinates"
            )
            if outer is None or not outer.text:
                continue
            fcoords = parse_coordinates_kml(outer.text)
            if len(fcoords) < 3:
                continue
            lonlat = [(lon, lat) for (lat, lon) in fcoords]
            try:
                shp = Polygon(lonlat)
                if not shp.is_valid or shp.is_empty:
                    continue
            except Exception:
                continue
            out.append({
                "name":            pname,
                "folium_coords":   fcoords,
                "shapely_polygon": shp,
                "desc_attrs":      attrs,
                "mod_n":           mod_n,
                "tur_n":           tur_n,
                "fundo_aq":        fundo_aq,   # 'AQ1', 'AQ2', o None
            })

    def _walk(node, inherited_mod_n=None, inherited_fundo_aq=None):
        for child in node:
            t = _tag(child)
            if t == "Folder":
                fname      = _text(child, "name")
                mod_n      = _extract_mod(fname)      or inherited_mod_n
                fundo_aq   = _extract_fundo_aq(fname) or inherited_fundo_aq
                _walk(child, mod_n, fundo_aq)
            elif t == "Placemark":
                _process_placemark(child, inherited_mod_n, inherited_fundo_aq)
            else:
                # Document, kml root, etc. — seguir bajando
                _walk(child, inherited_mod_n, inherited_fundo_aq)

    _walk(root)
    return out


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
# CARGA DE DATOS EXCEL
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

    # lat/lon son opcionales — filas sin coords se ubican por centroide KMZ
    mask_lat_ok = df["lat"].isna() | df["lat"].between(-90, 90)
    mask_lon_ok = df["lon"].isna() | df["lon"].between(-180, 180)
    df = df[mask_lat_ok & mask_lon_ok].copy()

    for c in ["empresa", "fundo", "modulo", "turno", "trampa", "lote", "tipo_trampa"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    return df


# ============================================================
# FILTRO DENTRO DEL KMZ — vectorizado
# ============================================================
def filter_within_union(df: pd.DataFrame, union_poly) -> pd.DataFrame:
    if union_poly is None or df.empty:
        return df

    # Filas sin coordenadas pasan siempre (se ubican por centroide KMZ)
    mask_no_coords = df["lat"].isna() | df["lon"].isna()
    df_no_coords   = df[mask_no_coords].copy()
    df_with_coords = df[~mask_no_coords].copy()

    if df_with_coords.empty:
        return df_no_coords

    minx, miny, maxx, maxy = union_poly.bounds
    mask_bbox = (
        df_with_coords["lon"].between(minx, maxx) &
        df_with_coords["lat"].between(miny, maxy)
    )
    df_bbox = df_with_coords[mask_bbox].copy()

    if df_bbox.empty:
        return pd.concat([df_no_coords], ignore_index=True)

    try:
        from shapely import contains_xy
        mask_exact = contains_xy(union_poly,
                                  df_bbox["lon"].values,
                                  df_bbox["lat"].values)
    except ImportError:
        _prep = prep(union_poly)
        mask_exact = [
            _prep.contains(Point(row.lon, row.lat))
            for row in df_bbox.itertuples()
        ]

    return pd.concat([df_no_coords, df_bbox[mask_exact]], ignore_index=True)


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
    t = min(max(float(val), 0.0), 3.0)
    if t < 0.5:
        return "#FFFFFF"
    amarillo = np.array([255, 255,   0], dtype=float)
    naranja  = np.array([255, 165,   0], dtype=float)
    rojo     = np.array([255,   0,   0], dtype=float)
    t_norm = (t - 0.5) / 2.5
    if t_norm <= 0.6:   color = amarillo + (naranja  - amarillo) * (t_norm / 0.6)
    else:               color = naranja  + (rojo     - naranja)  * ((t_norm - 0.6) / 0.4)
    return f"#{int(color[0]):02x}{int(color[1]):02x}{int(color[2]):02x}"

def _label_semaforo(cap: float) -> str:
    c = get_semaforo_category(cap)
    return {0: "🟢 Sin riesgo", 1: "🟡 Bajo", 2: "🟠 Medio", 3: "🔴 Alto"}[c]


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
    tip_lat, tip_lon = lat1, lon1
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
# (se alimenta de centroides de polígonos, no de grid)
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
# CARGA INICIAL — KMZ desde GitHub
# ============================================================
GITHUB_TOKEN_KMZ = st.secrets.get("GITHUB_TOKEN_KMZ", "")
KMZ_API_URL = (
    "https://api.github.com/repos/"
    "controloperacionalprize-boss/CAMPO_RENDIMIENTO/"
    "contents/MODULOS_PRIZE_PAIJAN.kmz"
)

@st.cache_data(show_spinner="Descargando KMZ desde GitHub…")
def download_kmz_from_github(api_url: str, token: str) -> bytes:
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
            f"Error descargando KMZ desde GitHub ({e.code}): {e.reason}."
        )
    except Exception as ex:
        raise RuntimeError(f"Error de red al descargar KMZ: {ex}")


# ── Cargar datos ────────────────────────────────────────────
df = load_trampas_anexadas()

try:
    kmz_bytes_raw = download_kmz_from_github(KMZ_API_URL, GITHUB_TOKEN_KMZ)
    kml_bytes     = extract_kml_from_kmz_bytes(kmz_bytes_raw)
    # v6: cargamos con descripción HTML parseada
    polygons      = load_polygons_with_desc(kml_bytes)
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


# Filtro KMZ vectorizado
if not dff.empty and union_polys is not None:
    dff = filter_within_union(dff, union_polys)

if not dff.empty:
    # Rellenar NaN en lote antes del groupby (NaN no es agrupable)
    if "lote" in dff.columns:
        dff["lote"] = dff["lote"].fillna("").astype(str).str.strip()
    else:
        dff["lote"] = ""
    # lat/lon pueden ser NaN (filas sin GPS) — rellenar con centinela para groupby
    dff["lat"] = dff["lat"].fillna(-9999.0)
    dff["lon"] = dff["lon"].fillna(-9999.0)
    dff = (
        dff
        .groupby(["lat", "lon", "fundo", "modulo", "turno", "trampa", "lote"], as_index=False)
        .agg({"capturas": "sum"})
    )
    # Restaurar NaN para coordenadas centinela
    mask_fake = (dff["lat"] == -9999.0) & (dff["lon"] == -9999.0)
    dff.loc[mask_fake, "lat"] = float("nan")
    dff.loc[mask_fake, "lon"] = float("nan")

if not dff.empty:
    dff["_cat"] = dff["capturas"].apply(get_semaforo_category)
    dff = dff[dff["_cat"].isin(cats_permitidas)].drop(columns=["_cat"])

st.sidebar.markdown("---")


# ============================================================
# MÉTODO DE INTERPOLACIÓN PARA CURVAS DE NIVEL
# ============================================================
metodo_interp = st.sidebar.radio(
    "🗺️ Método para curvas de nivel",
    options=[
        "GPS (si existe)",
        "Lotes KMZ (centroides)",
        "Híbrido (GPS + KMZ)"
    ],
    index=1,  # Por defecto: Lotes KMZ
    help="""
    • **GPS**: Solo puntos GPS reales (si hay ≥3)
    • **Lotes KMZ**: Centroide de cada lote KMZ
    • **Híbrido**: GPS si existe, sino centroide KMZ
    """
)

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
        num_niveles       = st.slider("Número de líneas de contorno",  2, 20, 10)
        grosor_lineas     = st.slider("Grosor de líneas",   1,  6, 3)
        mostrar_etiquetas = st.checkbox("Mostrar etiquetas de valor", value=False)
        opacidad_relleno  = st.slider("Opacidad de curvas", 0, 100, 98,
                                      help="Qué tan sólidas se ven las curvas. "
                                           "100 = opacas; bajo = semitransparentes.")

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
    if "fundo" not in df_in.columns or "turno" not in df_in.columns:
        return df_in
    def _vis(row):
        if float(row["capturas"]) > 2:
            return f"{row['fundo']}||{row['turno']}" in claves_visibles
        return True
    return df_in[df_in.apply(_vis, axis=1)].copy()

dff_map = apply_visibility(dff_activa)


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
    tiles="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
    attr='&copy; OpenStreetMap &copy; CARTO',
)

# Panes custom para que heatmap espectral / líneas de curvas queden ENCIMA
# de los marcadores (markerPane=600). zIndex=650.

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


# ============================================================
# PINTADO DE POLÍGONOS — v9
# Usar SIEMPRE dff_activa (datos completos sin filtro de visibilidad)
# para que todos los polígonos se pinten igual independientemente del método
# ============================================================

if not dff_activa.empty:
    dff_norm = dff_activa.copy()
    dff_norm["mod_n"]     = dff_norm["modulo"].apply(_norm_mod)
    dff_norm["tur_n"]     = dff_norm["turno"].apply(_norm_tur)
    dff_norm["lote_norm"] = dff_norm["lote"].apply(_norm_lote) if "lote" in dff_norm.columns else ""
    dff_norm = dff_norm.dropna(subset=["mod_n", "tur_n"])
    dff_norm["mod_n"] = dff_norm["mod_n"].astype(int)
    dff_norm["tur_n"] = dff_norm["tur_n"].astype(int)

    # ── Mapeo fundo Excel → código AQ del KMZ
    # Ajusta este dict si los nombres de fundo en el Excel cambian
    FUNDO_A_AQ = {
        "ARENA AZUL":   "AQ1",
        "QURI ALLPA":   "AQ1",
        "KAWSAY ALLPA": "AQ1",
        "AYLLU ALLPA":  "AQ1",
        "SANTA TERESA": "AQ2",
        "VIVADIS":      "AQ2",
    }
    def _fundo_to_aq(fundo_str: str) -> str | None:
        return FUNDO_A_AQ.get(str(fundo_str).strip().upper())

    dff_norm["fundo_aq"] = dff_norm["fundo"].apply(_fundo_to_aq)

    # ── Nivel 1: capturas por (fundo_aq, mod_n, tur_n, lote_norm)
    agg_lote = (
        dff_norm
        .groupby(["fundo_aq", "mod_n", "tur_n", "lote_norm"])["capturas"]
        .max()
        .reset_index()
        .rename(columns={"capturas": "cap_max"})
    )
    dict_lote = {
        (r["fundo_aq"], int(r["mod_n"]), int(r["tur_n"]), str(r["lote_norm"])): float(r["cap_max"])
        for _, r in agg_lote.iterrows()
    }

    # ── Nivel 2: capturas por (fundo_aq, mod_n, tur_n)
    agg_turno = (
        dff_norm
        .groupby(["fundo_aq", "mod_n", "tur_n"])
        .agg(n_lotes=("lote_norm", "nunique"), cap_max=("capturas", "max"))
        .reset_index()
    )
    dict_turno = {
        (r["fundo_aq"], int(r["mod_n"]), int(r["tur_n"])): (int(r["n_lotes"]), float(r["cap_max"]))
        for _, r in agg_turno.iterrows()
    }

    # ── Nivel 3: capturas por (fundo_aq, mod_n)
    agg_mod = (
        dff_norm
        .groupby(["fundo_aq", "mod_n"])
        .agg(n_lotes=("lote_norm", "nunique"), cap_max=("capturas", "max"))
        .reset_index()
    )
    dict_mod = {
        (r["fundo_aq"], int(r["mod_n"])): (int(r["n_lotes"]), float(r["cap_max"]))
        for _, r in agg_mod.iterrows()
    }
else:
    dict_lote  = {}
    dict_turno = {}
    dict_mod   = {}

# ── Construir unión de polígonos CON DATO (para enmascarar curvas de nivel)
_polys_con_dato = []
for _p in polygons:
    _mn      = _p.get("mod_n")
    _tn      = _p.get("tur_n")
    _faq     = _p.get("fundo_aq")
    _attrs   = _p.get("desc_attrs", {})
    _lot     = _norm_lote(_attrs.get("lote"))
    _cap     = None
    if _mn is not None and _tn is not None and _faq is not None:
        _key_l = (_faq, _mn, _tn, _lot) if _lot else None
        _key_t = (_faq, _mn, _tn)
        _key_m = (_faq, _mn)
        if _key_l and _key_l in dict_lote:
            _cap = dict_lote[_key_l]
        if _cap is None and _key_t in dict_turno:
            _cap = dict_turno[_key_t][1]
        if _cap is None and _key_m in dict_mod:
            _cap = dict_mod[_key_m][1]
    if _cap is not None:
        _polys_con_dato.append(_p["shapely_polygon"])

union_con_dato = unary_union(_polys_con_dato) if _polys_con_dato else None


# ============================================================
# HEATMAP + CURVAS DE NIVEL — imagen única inline (v2 style)
# ============================================================
GRID_RES = 1200  # Aumentado de 600 → 1200 para mejor resolución (menos pixelación)

grid_z_for_vectors = None
b64_curvas = None
grid_x_gv = grid_y_gv = mask_poly_gv = None

# Para el grid usamos solo filas con coordenadas válidas;
# los bounds se sacan de los polígonos KMZ si no hay suficientes puntos GPS
_dff_gps = dff_map.dropna(subset=["lat","lon"]) if not dff_map.empty else dff_map
_dff_gps = _dff_gps[(_dff_gps["lat"] != 0) | (_dff_gps["lon"] != 0)] if not _dff_gps.empty else _dff_gps

if (
    not dff_map.empty
    and union_polys is not None
    and dff_map["capturas"].sum() > 0
    and (len(_dff_gps) >= 3 or union_con_dato is not None)
):
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter

    # ============================================================
    # FUNCIÓN AUXILIAR PARA PREPARAR PUNTOS DE INTERPOLACIÓN
    # Soporta: GPS, Lotes KMZ, Híbrido
    # ============================================================
    def _prepare_interpolation_points(
        dff_gps, polygons, dict_turno, metodo="Híbrido"
    ):
        """
        Prepara puntos (lon, lat, capturas) para interpolación.
        
        Args:
            dff_gps: DataFrame con puntos GPS válidos
            polygons: Lista de polígonos KMZ
            dict_turno: Dict con datos por turno
            metodo: "GPS (si existe)", "Lotes KMZ (centroides)", "Híbrido (GPS + KMZ)"
        
        Returns:
            (x, y, z, msg): Arrays y mensaje informativo
        """
        import numpy as _np_int
        
        _pts_xy, _pts_z = [], []
        
        # ──── MÉTODO 1: GPS con agregación por turno (igual que KMZ)
        if metodo == "GPS (si existe)":
            # En lugar de interpolar puntos GPS individuales,
            # usamos la misma lógica que KMZ: agregamos por (fundo_aq, mod_n, tur_n)
            # pero priorizamos GPS cuando existen coordenadas reales
            
            _pts_xy_gps, _pts_z_gps = [], []
            _pts_set_gps = set()
            
            # Prioridad 1: Datos con GPS real (de Excel con coordenadas)
            if len(dff_gps) >= 1:
                for _, row in dff_gps.iterrows():
                    _pts_xy_gps.append((row["lon"], row["lat"]))
                    _pts_z_gps.append(row["capturas"])
                    _pts_set_gps.add((row["lon"], row["lat"]))
            
            # Fallback 2: Si GPS insuficiente, agregar KMZ centroides
            if len(_pts_xy_gps) < 3:
                for _pp in polygons:
                    _mn2  = _pp.get("mod_n")
                    _tn2  = _pp.get("tur_n")
                    _faq2 = _pp.get("fundo_aq")
                    if _mn2 is None or _tn2 is None or _faq2 is None:
                        continue
                    _key_t2 = (_faq2, _mn2, _tn2)
                    if _key_t2 in dict_turno:
                        _c2 = _pp["shapely_polygon"].centroid
                        xy_tuple = (_c2.x, _c2.y)
                        if xy_tuple not in _pts_set_gps:  # No duplicar
                            _pts_xy_gps.append(xy_tuple)
                            _pts_z_gps.append(dict_turno[_key_t2][1])
            
            # Retornar si hay suficientes puntos
            if len(_pts_xy_gps) >= 3:
                _arr = _np_int.array(_pts_xy_gps)
                x, y, z = _arr[:,0], _arr[:,1], _np_int.array(_pts_z_gps, dtype=float)
                msg = f"📍 GPS ({len(dff_gps)}) + 🗺️ KMZ ({len(_pts_xy_gps) - len(dff_gps)}) = {len(x)} total"
                return x, y, z, msg
            else:
                return None, None, None, f"❌ Solo {len(_pts_xy_gps)} puntos (mín 3)"
        
        # ──── MÉTODO 2: KMZ centroides
        elif metodo == "Lotes KMZ (centroides)":
            for _pp in polygons:
                _mn2  = _pp.get("mod_n")
                _tn2  = _pp.get("tur_n")
                _faq2 = _pp.get("fundo_aq")
                if _mn2 is None or _tn2 is None or _faq2 is None:
                    continue
                _key_t2 = (_faq2, _mn2, _tn2)
                if _key_t2 in dict_turno:
                    _c2 = _pp["shapely_polygon"].centroid
                    _pts_xy.append((_c2.x, _c2.y))
                    _pts_z.append(dict_turno[_key_t2][1])
            
            if len(_pts_xy) >= 3:
                _arr = _np_int.array(_pts_xy)
                x, y, z = _arr[:,0], _arr[:,1], _np_int.array(_pts_z, dtype=float)
                return x, y, z, f"🗺️ KMZ ({len(x)} lotes)"
            else:
                return None, None, None, f"❌ Solo {len(_pts_xy)} KMZ"
        
        # ──── MÉTODO 3: HÍBRIDO (GPS + KMZ)
        elif metodo == "Híbrido (GPS + KMZ)":
            n_gps = 0
            # Primero: GPS si existe
            if len(dff_gps) >= 1:
                for _, row in dff_gps.iterrows():
                    _pts_xy.append((row["lon"], row["lat"]))
                    _pts_z.append(row["capturas"])
                    n_gps += 1
            
            # Segundo: Llenar con KMZ donde falte
            _pts_set = set((xy[0], xy[1]) for xy in _pts_xy)
            n_kmz = 0
            
            for _pp in polygons:
                _mn2  = _pp.get("mod_n")
                _tn2  = _pp.get("tur_n")
                _faq2 = _pp.get("fundo_aq")
                if _mn2 is None or _tn2 is None or _faq2 is None:
                    continue
                _key_t2 = (_faq2, _mn2, _tn2)
                if _key_t2 in dict_turno:
                    _c2 = _pp["shapely_polygon"].centroid
                    xy_tuple = (_c2.x, _c2.y)
                    if xy_tuple not in _pts_set:
                        _pts_xy.append(xy_tuple)
                        _pts_z.append(dict_turno[_key_t2][1])
                        n_kmz += 1
            
            if len(_pts_xy) >= 3:
                _arr = _np_int.array(_pts_xy)
                x, y, z = _arr[:,0], _arr[:,1], _np_int.array(_pts_z, dtype=float)
                msg = f"🔀 Híbrido: {n_gps} GPS + {n_kmz} KMZ = {len(x)} total"
                return x, y, z, msg
            else:
                return None, None, None, f"❌ Solo {len(_pts_xy)} puntos"
        
        return None, None, None, "❌ Método desconocido"
    
    # ──── APLICAR LÓGICA DE SELECCIÓN ────
    if len(_dff_gps) >= 3 and metodo_interp == "GPS (si existe)":
        x, y, z, msg_interp = _prepare_interpolation_points(
            _dff_gps, polygons, dict_turno, metodo="GPS (si existe)"
        )
    else:
        # Usar KMZ o Híbrido
        x, y, z, msg_interp = _prepare_interpolation_points(
            _dff_gps, polygons, dict_turno, metodo=metodo_interp
        )
    
    # Mostrar info del método usado en curvas de nivel
    if modo_color == "Curvas de Nivel" and x is not None and msg_interp:
        st.sidebar.info(msg_interp)

    if x is None or len(x) < 3:
        grid_z = None
        xmin = xmax = ymin = ymax = 0
        grid_x = grid_y = None
    else:
        max_capturas = max(z.max(), 1)
        eps          = 1e-6
        if union_polys is not None:
            _bx1, _by1, _bx2, _by2 = union_polys.bounds
            xmin, xmax = min(x.min(), _bx1), max(x.max(), _bx2)
            ymin, ymax = min(y.min(), _by1), max(y.max(), _by2)
        else:
            xmin, xmax = x.min(), x.max()
            ymin, ymax = y.min(), y.max()
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
        try:
            from shapely import contains_xy as _contains_xy
            mask_poly = _contains_xy(
                union_polys,
                grid_x.ravel(),
                grid_y.ravel()
            ).reshape(grid_z.shape)
        except ImportError:
            _prep2 = prep(union_polys)
            pts_flat = np.vstack((grid_x.ravel(), grid_y.ravel())).T
            mask_poly = np.array(
                [_prep2.contains(Point(lx, ly)) for lx, ly in pts_flat]
            ).reshape(grid_z.shape)

        grid_z_masked = grid_z.copy()
        grid_z_masked[~mask_poly] = np.nan

        # ============================================================
        # CALCULAR mask_dato una sola vez (disponible para todos modos)
        # ============================================================
        mask_dato = None
        if union_con_dato is not None:
            try:
                from shapely import contains_xy as _cxy
                mask_dato = _cxy(
                    union_con_dato,
                    grid_x.ravel(), grid_y.ravel()
                ).reshape(grid_z_masked.shape)
            except ImportError:
                _prep_dato = prep(union_con_dato)
                pts_flat_d = np.vstack((grid_x.ravel(), grid_y.ravel())).T
                mask_dato  = np.array(
                    [_prep_dato.contains(Point(lx, ly)) for lx, ly in pts_flat_d]
                ).reshape(grid_z_masked.shape)

        gz_smooth_global             = gaussian_filter(
            np.where(np.isnan(grid_z_masked), 0, grid_z_masked), sigma=8
        )
        gz_smooth_global[~mask_poly] = np.nan
        grid_z_for_vectors           = gz_smooth_global
        grid_x_gv, grid_y_gv, mask_poly_gv = grid_x, grid_y, mask_poly

        if not np.all(np.isnan(grid_z_masked)):

            if modo_color == "Curvas de Nivel":
                # ── CURVAS DE NIVEL SUAVIZADAS: Grilla de alta resolución + Suavizado progresivo
                # SUAVIZADO PROGRESIVO: Aplicar Gaussian múltiples veces para máxima suavidad
                gz_smooth = np.where(np.isnan(grid_z_masked), 0, grid_z_masked).copy()
                
                # Primera pasada: Suavizado moderado
                gz_smooth = gaussian_filter(gz_smooth, sigma=4.5)
                # Segunda pasada: Suavizado adicional para eliminar artefactos
                gz_smooth = gaussian_filter(gz_smooth, sigma=2.0)
                # Tercera pasada: Refinamiento final
                gz_smooth = gaussian_filter(gz_smooth, sigma=1.0)
                
                gz_smooth[~mask_poly] = np.nan

                if mask_dato is not None:
                    gz_smooth[~mask_dato] = np.nan
                    gz_smooth_masked_cv = np.ma.masked_where(~mask_dato, gz_smooth)
                else:
                    gz_smooth_masked_cv = np.ma.masked_where(~mask_poly, gz_smooth)

                max_level = max(4.0, float(max_capturas) + 1)

                fig2, ax2 = plt.subplots(figsize=(GRID_RES/100, GRID_RES/100), dpi=150)
                ax2.set_axis_off()
                fig2.patch.set_alpha(0)
                try:
                    # BANDAS DE RELLENO: Muchos niveles para máxima suavidad
                    num_niveles_relleno = 25  # Aumentado de 15 → 25 para transiciones ultrasaves
                    niveles_relleno = np.linspace(0.01, max_level, num_niveles_relleno)
                    
                    # Paleta de colores suavizada: Verde → Amarillo → Naranja → Rojo
                    from matplotlib.colors import LinearSegmentedColormap
                    colores_paleta = ["#00FF00", "#CCFF00", "#FFFF00", "#FFCC00", "#FFA500", "#FF7500", "#FF6600", "#FF0000"]
                    cmap_custom = LinearSegmentedColormap.from_list("mosca_fruta", colores_paleta)
                    
                    contourf_obj = ax2.contourf(
                        grid_x.T, grid_y.T, gz_smooth_masked_cv.T,
                        levels=niveles_relleno,
                        cmap=cmap_custom,
                        alpha=1.0,
                        extend="max"
                    )
                    
                    # LÍNEAS DE CONTORNO: Delgadas pero claras
                    if grosor_lineas and grosor_lineas > 0:
                        # Niveles de línea densos para mejor definición
                        if num_niveles and num_niveles > 0:
                            niveles_linea = np.linspace(0.1, float(max_capturas), num_niveles).tolist()
                        else:
                            # Niveles por defecto densos
                            niveles_linea = np.linspace(0.1, float(max_capturas), 15).tolist()
                        
                        if not niveles_linea:
                            niveles_linea = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
                        
                        cl2 = ax2.contour(
                            grid_x.T, grid_y.T, gz_smooth_masked_cv.T,
                            levels=niveles_linea,
                            colors="#1a1a1a",  # Negro profundo
                            linewidths=grosor_lineas * 0.5,  # ← Líneas más delgadas (0.5x en lugar de 0.8x)
                            alpha=0.90,
                            linestyles="solid"
                        )
                        
                        # Etiquetas si está activado
                        if mostrar_etiquetas:
                            ax2.clabel(cl2, inline=True, fontsize=6,
                                       fmt="%.1f", colors="#000000", 
                                       bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
                except Exception as e:
                    st.warning(f"Error en curvas de nivel: {e}")
                
                ax2.set_xlim(xmin, xmax)
                ax2.set_ylim(ymin, ymax)
                plt.tight_layout(pad=0)
                buf2 = io.BytesIO()
                fig2.savefig(buf2, format="PNG", bbox_inches="tight",
                             pad_inches=0, transparent=True, dpi=200)
                plt.close(fig2)
                buf2.seek(0)
                img2 = PILImage.open(buf2).convert("RGBA")
                buf3 = io.BytesIO()
                img2.save(buf3, format="PNG")
                buf3.seek(0)
                b64_curvas = base64.b64encode(buf3.read()).decode()
                # Se añade al mapa DESPUÉS de los polígonos y marcadores (ver abajo)

            else:
                # ESPECTRAL / NORMAL — Suavizado multi-pasada + Alta resolución
                # Aplicar suavizado progresivo (igual que Curvas de Nivel)
                gz_spectral = np.where(np.isnan(grid_z_masked), 0, grid_z_masked).copy()
                
                # Primera pasada: Suavizado fuerte
                gz_spectral = gaussian_filter(gz_spectral, sigma=4.5)
                # Segunda pasada: Suavizado mediano
                gz_spectral = gaussian_filter(gz_spectral, sigma=2.0)
                # Tercera pasada: Refinamiento
                gz_spectral = gaussian_filter(gz_spectral, sigma=1.0)
                
                gz_spectral[~mask_poly] = np.nan
                
                # Crear raster RGBA con suavizado aplicado
                rgba  = np.zeros(gz_spectral.shape + (4,), dtype=np.uint8)
                valid = ~np.isnan(gz_spectral)
                
                # Usar mask_dato si existe (solo polígonos con datos)
                mask_to_use = mask_dato if union_con_dato is not None else mask_poly
                
                # Paleta espectral mejorada: Verde → Amarillo → Naranja → Rojo
                from matplotlib.colors import LinearSegmentedColormap
                colores_espectral = ["#00FF00", "#CCFF00", "#FFFF00", "#FFCC00", 
                                    "#FFA500", "#FF7500", "#FF6600", "#FF0000"]
                cmap_espectral = LinearSegmentedColormap.from_list("espectral_mosca", colores_espectral)
                
                # Normalizar valores para usar con el colormap
                max_val = float(max_capturas) if max_capturas > 0 else 1.0
                
                for i in range(gz_spectral.shape[0]):
                    for j in range(gz_spectral.shape[1]):
                        # Solo pintar si: hay valor válido Y está dentro del área con datos
                        if valid[i, j] and mask_to_use[i, j]:
                            val = gz_spectral[i, j]
                            
                            # Usar colormap lineal en lugar de función discreta
                            if modo_color == "Espectral":
                                # Normalizar a [0, 1]
                                val_norm = min(max(val / max_val, 0), 1.0)
                                # Obtener color del colormap
                                rgba_color = cmap_espectral(val_norm)
                                rgba[i, j, :3] = (np.array(rgba_color[:3]) * 255).astype(np.uint8)
                                # Opacidad basada en valor
                                rgba[i, j, 3] = int(255 * (0.3 + 0.7 * val_norm))  # 30-100% opaco
                            else:
                                # Modo Normal: usar colores semáforo
                                ch = get_color_normal(val)
                                rgb = mcolors.to_rgb(ch)
                                rgba[i, j, :3] = (np.array(rgb) * 255).astype(np.uint8)
                                rgba[i, j, 3] = int(255 * (0.4 + 0.6 * (val / max_val)))
                
                # Limpiar píxeles fuera del área de datos
                rgba[~mask_to_use] = [0, 0, 0, 0]

                pil_img = PILImage.fromarray(
                    np.flipud(rgba.transpose(1, 0, 2)), mode="RGBA"
                )
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                pil_img.save(tmp.name, format="PNG")
                tmp.close()
                with open(tmp.name, "rb") as f_tmp:
                    b64_heat = base64.b64encode(f_tmp.read()).decode()
                os.unlink(tmp.name)
                folium.raster_layers.ImageOverlay(
                    image="data:image/png;base64," + b64_heat,
                    bounds=[[ymin, xmin], [ymax, xmax]],
                    opacity=0.98, name=f"Heatmap ({modo_color})",  # ← 0.98 para máxima visibilidad
                ).add_to(m)

# ============================================================
# POLÍGONOS COLOREADOS — DESPUÉS (z-index alto)
# ============================================================
fg_poly_fill = FeatureGroup(name="🗺️ Módulos coloreados", show=True)

for p in polygons:
    mod_n    = p.get("mod_n")
    tur_n    = p.get("tur_n")
    fundo_aq = p.get("fundo_aq")   # 'AQ1', 'AQ2', o None
    attrs    = p.get("desc_attrs", {})
    lote_kmz = _norm_lote(attrs.get("lote"))

    tooltip_lines = [f"<b>{escape(p['name'])}</b>"]
    if fundo_aq:
        tooltip_lines.append(f"<b>Fundo KMZ:</b> {escape(fundo_aq)}")
    for k, v in attrs.items():
        if k not in ("fid", "id") and v:
            tooltip_lines.append(f"<b>{k.capitalize()}:</b> {escape(str(v))}")

    cap_max   = None
    nivel_txt = ""

    if mod_n is not None and tur_n is not None and fundo_aq is not None:
        key_l = (fundo_aq, mod_n, tur_n, lote_kmz) if lote_kmz else None
        key_t = (fundo_aq, mod_n, tur_n)
        key_m = (fundo_aq, mod_n)

        # Nivel 1 — lote exacto
        if key_l and key_l in dict_lote:
            cap_max   = dict_lote[key_l]
            nivel_txt = f"Lote {lote_kmz}"

        # Nivel 2 — turno con dato en Excel → pintar todos sus polígonos
        if cap_max is None and key_t in dict_turno:
            cap_max   = dict_turno[key_t][1]
            nivel_txt = f"Turno {tur_n}"

        # Nivel 3 — módulo con dato en Excel → expandir a todo el módulo
        if cap_max is None and key_m in dict_mod:
            cap_max   = dict_mod[key_m][1]
            nivel_txt = f"Módulo {mod_n}"

    if cap_max is not None:
        # Los polígonos SIEMPRE usan el color semafórico normal como capa base
        # dominante en los tres modos. Los overlays raster van encima.
        fill_color  = get_color_normal(cap_max)
        import matplotlib.colors as _mc
        r, g, b     = _mc.to_rgb(fill_color)
        border_color = "#{:02x}{:02x}{:02x}".format(
            max(0, int(r*255) - 60),
            max(0, int(g*255) - 60),
            max(0, int(b*255) - 60),
        )
        # Relleno semafórico 0.25 en TODOS los modos
        fill_op = 0.65
        weight   = 3  # ← Aumentado de 2 a 3 para que se vean mejor
        tooltip_lines.append(f"<b>Capturas:</b> {int(cap_max)}")
        tooltip_lines.append(f"<b>Nivel:</b> {_label_semaforo(cap_max)}")
        tooltip_lines.append(f"<b>Origen dato:</b> {nivel_txt}")
    else:
        fill_color   = "#888888"
        border_color = "#555555"
        fill_op  = 0.0
        weight   = 1
        tooltip_lines.append("<i style='color:#999;'>Sin dato en filtro actual</i>")

    # Ajustar opacidad según el modo de visualización
    if modo_color == "Curvas de Nivel":
        # En Curvas de Nivel: aumentar opacidad de polígonos
        fill_op_final = (fill_op * 1.3 if fill_op > 0 else 0)
    elif modo_color == "Espectral":
        # En Espectral: REDUCIR opacidad para no opacar el heatmap
        fill_op_final = (fill_op * 0.4 if fill_op > 0 else 0)  # Reduce a 40% de la opacidad original
    else:
        # En Normal: mantener opacidad original
        fill_op_final = fill_op
    
    fill_op_final = min(fill_op_final, 1.0)  # Clamped a máximo 1.0

    folium.Polygon(
        locations=p["folium_coords"],
        tooltip=folium.Tooltip("<br>".join(tooltip_lines)),
        color=border_color,      # borde siempre visible y oscuro
        weight=weight,
        fill=True,
        fill_color=fill_color,   # relleno con opacidad controlable
        fill_opacity=fill_op_final,
        opacity=0.95,             # borde siempre al 100%
    ).add_to(fg_poly_fill)

fg_poly_fill.add_to(m)


# ============================================================
# MARCADORES DE TRAMPAS — v9
# Posición: centroide del polígono KMZ exacto del LOTE
#   Cruce: (mod_n, tur_n, lote_norm) Excel ↔ KMZ
#   Un marcador por cada (mod_n, tur_n, lote_norm) único
#   Capturas: máximo del grupo
#
# PINTADO DE FONDO: a nivel de TURNO (todos los lotes del
#   mismo turno reciben el mismo color) — ya hecho arriba.
# ============================================================
if not dff_map.empty and polygons:
    # ── Índice KMZ: (fundo_aq, mod_n, tur_n, lote_norm) → shapely_polygon
    kmz_lote_index  = {}   # {(fundo_aq, mod_n, tur_n, lote_str): shapely_polygon}
    kmz_turno_index = {}   # {(fundo_aq, mod_n, tur_n): [shapely_polygon, ...]}

    for p in polygons:
        mn  = p.get("mod_n")
        tn  = p.get("tur_n")
        faq = p.get("fundo_aq")
        lot = _norm_lote(p.get("desc_attrs", {}).get("lote"))
        if mn is None or tn is None or faq is None:
            continue
        key_t = (faq, int(mn), int(tn))
        kmz_turno_index.setdefault(key_t, []).append(p["shapely_polygon"])
        if lot:
            key_l = (faq, int(mn), int(tn), lot)
            kmz_lote_index[key_l] = p["shapely_polygon"]

    # ── Normalizar Excel
    dff_markers = dff_map.copy()
    dff_markers["mod_n"]     = dff_markers["modulo"].apply(_norm_mod)
    dff_markers["tur_n"]     = dff_markers["turno"].apply(_norm_tur)
    dff_markers["lote_norm"] = dff_markers["lote"].apply(_norm_lote) if "lote" in dff_markers.columns else None
    dff_markers["fundo_aq"]  = dff_markers["fundo"].apply(_fundo_to_aq)
    dff_markers = dff_markers.dropna(subset=["mod_n", "tur_n", "fundo_aq"])
    dff_markers["mod_n"] = dff_markers["mod_n"].astype(int)
    dff_markers["tur_n"] = dff_markers["tur_n"].astype(int)

    # ── Agrupar: un marcador por (fundo_aq, mod_n, tur_n, lote_norm)
    group_cols = ["fundo_aq", "mod_n", "tur_n", "lote_norm", "fundo", "modulo", "turno"]
    group_cols = [c for c in group_cols if c in dff_markers.columns]
    agg_markers = (
        dff_markers
        .groupby(group_cols, as_index=False)
        .agg({"capturas": "max"})
    )

    fg_pts = FeatureGroup(name="📍 Trampas", show=True)

    for _, r in agg_markers.iterrows():
        faq  = r.get("fundo_aq")
        mn   = int(r["mod_n"])
        tn   = int(r["tur_n"])
        lot  = str(r.get("lote_norm", "")) if r.get("lote_norm") else None
        cap  = float(r["capturas"])

        # Buscar polígono exacto del lote primero,
        # si no existe usar centroide del turno completo
        poly_geom = None
        if lot and faq:
            poly_geom = kmz_lote_index.get((faq, mn, tn, lot))
        if poly_geom is None and faq:
            polys_t = kmz_turno_index.get((faq, mn, tn))
            if polys_t:
                poly_geom = unary_union(polys_t) if len(polys_t) > 1 else polys_t[0]

        if poly_geom is None:
            continue  # sin polígono KMZ → no mostrar

        centroid = poly_geom.centroid
        c_lat    = centroid.y
        c_lon    = centroid.x

        color     = (get_color_espectral(cap)
                     if modo_color == "Espectral"
                     else get_color_normal(cap))
        turno_txt = escape(str(r.get("turno", "")))
        lote_txt  = escape(str(lot)) if lot else "—"

        tip = (
            f"<b>Fundo:</b> {escape(str(r.get('fundo', '')))}<br>"
            f"<b>Módulo:</b> {escape(str(r.get('modulo', '')))}<br>"
            f"<b>Turno:</b> {turno_txt}<br>"
            f"<b>Lote:</b> {lote_txt}<br>"
            f"<b>Capturas máx:</b> {int(cap)}<br>"
            f"<b>Nivel:</b> {_label_semaforo(cap)}"
        )

        folium.CircleMarker(
            location=(c_lat, c_lon), radius=2,
            color="black", weight=1, fill=True,
            fill_color=color, fill_opacity=0.9,
            tooltip=tip
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
            location=(c_lat, c_lon),
            icon=folium.DivIcon(
                html=label_html, icon_size=(0, 0), icon_anchor=(0, -12)
            )
        ).add_to(fg_pts)

    fg_pts.add_to(m)

# ── Curvas de nivel ENCIMA de polígonos (mejor visibilidad)
if modo_color == "Curvas de Nivel" and b64_curvas:
    # Usar opacidad del slider si existe, sino usar 0.98 por defecto
    opacity_curvas = (opacidad_relleno / 100.0) if opacidad_relleno is not None else 0.98
    contour_layer = folium.raster_layers.ImageOverlay(
        image="data:image/png;base64," + b64_curvas,
        bounds=[[ymin, xmin], [ymax, xmax]],
        opacity=opacity_curvas,  # ← Controlable por slider, 98% por defecto
        name="Curvas de nivel",
        className="contour-overlay"
    )
    contour_layer.add_to(m)
    
    # CSS para poner curvas ENCIMA pero con z-index apropiado
    m.get_root().html.add_child(folium.Element("""
    <style>
        /* Curvas de nivel encima de los polígonos */
        .leaflet-pane.leaflet-overlay-pane {
            z-index: 410 !important;  /* Un poco arriba de polígonos */
        }
        .contour-overlay {
            z-index: 410 !important;
        }
        .leaflet-image-layer {
            z-index: 410 !important;
        }
    </style>
    """))


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
        border-radius:3px;"></span>&gt; 2 capturas<br><br>
  <span style="display:inline-block;width:18px;height:18px;background:#888888;
        border:1px solid #333;vertical-align:middle;margin-right:8px;
        border-radius:3px;"></span>Sin dato
</div>
"""))


# ============================================================
# VECTORES DE PROPAGACIÓN — DIBUJAR FLECHAS
# ============================================================
if mostrar_vectores and grid_z_for_vectors is not None and grid_x_gv is not None:
    try:
        vectors = compute_gradient_vectors(
            grid_x_gv, grid_y_gv, grid_z_for_vectors, mask_poly_gv,
            n_arrows=n_arrows, min_magnitude=min_mag
        )
        
        if vectors:
            fg_vectors = FeatureGroup(name="🧭 Vectores de Propagación", show=True)
            
            escala_flecha_m = escala_flecha / 10000.0
            head_size_deg   = head_size / 100000.0
            
            for v in vectors:
                lat0 = v["lat"]
                lon0 = v["lon"]
                dlat = v["dlat"]
                dlon = v["dlon"]
                mag  = v["magnitude"]
                
                lat1 = lat0 + dlat * mag * escala_flecha_m
                lon1 = lon0 + dlon * mag * escala_flecha_m
                
                tooltip_txt = f"Magnitud: {mag:.3f}"
                
                draw_arrow(
                    fg_vectors, lat0, lon0, lat1, lon1,
                    color=color_flechas, weight=2, opacity=0.7,
                    head_size_deg=head_size_deg, tooltip=tooltip_txt
                )
            
            fg_vectors.add_to(m)
    except Exception as e:
        st.warning(f"Error dibujando vectores de propagación: {e}")


# ============================================================
# CONTROL DE CAPAS
# ============================================================
folium.LayerControl(collapsed=False).add_to(m)


# ============================================================
# NOMBRE DE ARCHIVO PNG / HTML
# ============================================================
_partes_nombre = []
if sel_anio:   _partes_nombre.append("A" + "-".join(map(str, sel_anio)))
if sel_semana: _partes_nombre.append("S" + "-".join(map(str, sel_semana)))
if sel_fundo:  _partes_nombre.append("F" + "-".join(sel_fundo))
if sel_mod:    _partes_nombre.append("M" + "-".join(sel_mod))
if sel_lote:   _partes_nombre.append("L" + "-".join(sel_lote))
if sel_turno:  _partes_nombre.append("T" + "-".join(sel_turno))
_nombre_png = ("MosaicaFruta_" + "_".join(_partes_nombre) if _partes_nombre
               else "MosaicaFruta_TodosLosDatos")
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
            useCORS: true, allowTaint: false,
            backgroundColor: '#ffffff', scale: 2,
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
# ============================================================
_html_mapa   = m._repr_html_()
_nombre_html = _nombre_png + ".html"

GITHUB_TOKEN  = st.secrets.get("GITHUB_TOKEN",  "")
GITHUB_OWNER  = st.secrets.get("GITHUB_OWNER",  "controloperacionalprize-boss")
GITHUB_REPO   = st.secrets.get("GITHUB_REPO",   "mapa_html")
GITHUB_BRANCH = st.secrets.get("GITHUB_BRANCH", "main")
GITHUB_FILE   = "mapa_mosca.html"

GITHUB_PAGES_URL = (
    f"https://{GITHUB_OWNER}.github.io/{GITHUB_REPO}/{GITHUB_FILE}"
)

def _push_file_github(api_url, html_content, branch, mensaje, headers):
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

def _subir_a_github(html_content):
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
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    if sel_semana and sel_anio:
        sufijo = f"A{'-'.join(map(str,sorted(sel_anio)))}_S{'-'.join(map(str,sorted(sel_semana)))}"
    elif sel_anio:
        sufijo = "A" + "-".join(map(str, sorted(sel_anio)))
    elif sel_semana:
        sufijo = "S" + "-".join(map(str, sorted(sel_semana)))
    else:
        sufijo = "SinFiltro"
    nombre_h = f"historico/mapa_{sufijo}.html"
    mensaje  = f"Mapa actualizado {ts} | {sufijo}"
    ok1, res1 = _push_file_github(
        f"{base_repo}/{GITHUB_FILE}", html_content, GITHUB_BRANCH, mensaje, headers
    )
    if not ok1:
        return False, f"Error subiendo archivo fijo: {res1}"
    ok2, res2 = _push_file_github(
        f"{base_repo}/{nombre_h}", html_content, GITHUB_BRANCH, mensaje, headers
    )
    url_historico = f"https://{GITHUB_OWNER}.github.io/{GITHUB_REPO}/{nombre_h}"
    if ok2:
        return True, (res1, url_historico)
    else:
        return True, (res1, f"Histórico falló: {res2}")


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


# ── UI Publicar ─────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### 🌐 Publicar")
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
                import platform, time, tempfile, os
                tmp_html = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
                tmp_html.write(_html_mapa.encode("utf-8"))
                tmp_html.close()
                opts = Options()
                opts.add_argument("--headless=new")
                opts.add_argument("--no-sandbox")
                opts.add_argument("--disable-dev-shm-usage")
                opts.add_argument("--disable-gpu")
                opts.add_argument("--window-size=1920,1080")
                if platform.system() == "Windows":
                    from webdriver_manager.chrome import ChromeDriverManager
                    service = Service(ChromeDriverManager().install())
                else:
                    opts.binary_location = "/usr/bin/chromium"
                    service = Service("/usr/bin/chromedriver")
                driver = webdriver.Chrome(service=service, options=opts)
                driver.get(f"file:///{tmp_html.name}")
                time.sleep(5)
                total_width  = driver.execute_script("return document.body.scrollWidth")
                total_height = driver.execute_script("return document.body.scrollHeight")
                driver.set_window_size(total_width, total_height)
                time.sleep(2)
                driver.execute_script("window.scrollTo(0, 0);")
                time.sleep(1)
                png_bytes = driver.get_screenshot_as_png()
                st.sidebar.download_button(
                    label="⬇️ Descargar PNG", data=png_bytes,
                    file_name=f"{_nombre_png}.png", mime="image/png",
                    key="btn_dl_png_real"
                )
                st.sidebar.success("✅ PNG listo")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")
            finally:
                try:
                    if driver: driver.quit()
                except: pass
                try:
                    if tmp_html: os.unlink(tmp_html.name)
                except: pass
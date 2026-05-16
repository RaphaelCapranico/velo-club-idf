
import io
import math
from functools import lru_cache
from random import sample

import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import requests

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment

from backend import (
    load_graph, load_club_traces, route, route_waypoints,
    similarity_to_club, f1_geo,
    export_gpx_string, read_gpx_string,
)


# ============================================================================
# CONFIG
# ============================================================================

st.set_page_config(
    page_title="Vélo Plaisir IdF",
    page_icon="🚴",
    layout="wide",
)

st.title("🚴 Planificateur d'itinéraires vélo — Île-de-France")
st.caption("Routes plaisir")

# ============================================================================
# SESSION STATE
# ============================================================================

def init_session_state():

    defaults = {
        "waypoints": [],
        "route_result": None,
        "uploaded_trace": None,
        "uploaded_elevs": None,
        "uploaded_sim": None,
        "comparison": None,
        "bumps_route": None,
        "bumps_uploaded": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v



@st.cache_resource(show_spinner="Chargement du graphe routier...")
def cached_graph():
    return load_graph()


@st.cache_resource(show_spinner="Chargement des traces club...")
def cached_traces():
    return load_club_traces()


@st.cache_data
def cached_simplified_traces(_traces):

    simplified = []
    for t in _traces:
        coords = t["coords"]
        n = len(coords)
        if n <= 100:
            simp = coords
        else:
            step = max(1, n // 100)
            simp = coords[::step]
            if simp[-1] != coords[-1]:
                simp.append(coords[-1])
        simplified.append({"name": t["name"], "coords": simp})
    return simplified



# ============================================================================
# GÉOCODAGE
# ============================================================================

@lru_cache(maxsize=200)
def geocode_address(address):

    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": address,
            "format": "json",
            "limit": 1,
            "countrycodes": "fr",
            "viewbox": "1.2,49.3,3.6,48.0",
            "bounded": 1,
        }
        headers = {"User-Agent": "VeloClubIDF/1.0"}
        resp = requests.get(url, params=params, headers=headers, timeout=5)
        results = resp.json()
        if results:
            return float(results[0]["lat"]), float(results[0]["lon"]), results[0]["display_name"]
    except Exception as e:
        print(f"Geocoding error: {e}")
    return None


# ============================================================================
# DÉTECTION DES BOSSES
# ============================================================================

def haversine_m(p1, p2):
    R = 6371000
    phi1, phi2 = math.radians(p1[0]), math.radians(p2[0])
    dphi = math.radians(p2[0] - p1[0])
    dlam = math.radians(p2[1] - p1[1])
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))


def _resample_elevation(coords, elevations, step_m=25):

    if len(coords) < 2 or len(elevations) != len(coords):
        return [], []
    cum = [0.0]
    for i in range(1, len(coords)):
        cum.append(cum[-1] + haversine_m(coords[i-1], coords[i]))
    total = cum[-1]
    dists, elevs = [], []
    d = 0.0
    j = 0
    while d <= total:
        while j < len(cum) - 1 and cum[j+1] < d:
            j += 1
        if j >= len(cum) - 1:
            e = elevations[-1]
        else:
            denom = max(cum[j+1] - cum[j], 0.001)
            frac = (d - cum[j]) / denom
            e = elevations[j] + frac * (elevations[j+1] - elevations[j])
        dists.append(d)
        elevs.append(e)
        d += step_m
    return dists, elevs


def _smooth(vals, window):
    out = []
    half = window // 2
    n = len(vals)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out.append(sum(vals[lo:hi]) / (hi - lo))
    return out


def _solve_climb_speed(power_w, grade_pct, mass_kg=77, CdA=0.30, Crr=0.004):
    """Résout P = (m·g·(grade+Crr))·v + 0.5·rho·CdA·v^3 par Newton."""
    g, rho = 9.81, 1.225
    grade = grade_pct / 100
    F_lin = mass_kg * g * (grade + Crr)
    F_aero = 0.5 * rho * CdA
    v = 5.0
    for _ in range(50):
        f = F_lin * v + F_aero * v**3 - power_w
        df = F_lin + 3 * F_aero * v**2
        v_new = v - f / df
        if abs(v_new - v) < 0.0001:
            v = v_new
            break
        v = max(0.5, v_new)
    return v


def _format_time(sec):
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m}'{s:02d}" + '"'


def _find_pointu_section(smooth, dists, start_idx, peak_idx, bump_grade,
                          min_grade=6.5, min_delta=1.5, min_length=150):
    """Trouve la section avec la pente la plus forte qui dépasse les seuils.

    Critères stricts :
    - Pente >= 6.5%
    - Pente - pente_bosse >= 1.5 point (vraiment plus pentue que le reste)
    - Longueur >= 150m
    """
    step = dists[1] - dists[0] if len(dists) > 1 else 25
    bump_length = dists[peak_idx] - dists[start_idx]
    max_window = min(bump_length * 0.85, 500)

    best = None
    for window_m in range(int(min_length), int(max_window) + 25, 25):
        ws = max(2, int(window_m / step))
        if ws > peak_idx - start_idx:
            continue

        # Trouve la position avec la pente max pour cette fenêtre
        best_pos_grade = -1
        best_pos_i = None
        for i in range(start_idx, peak_idx - ws + 1):
            j = i + ws
            length = dists[j] - dists[i]
            gain = smooth[j] - smooth[i]
            if gain <= 0:
                continue
            grade = gain / length * 100
            if grade > best_pos_grade:
                best_pos_grade = grade
                best_pos_i = i
                best_pos_length = length

        if best_pos_i is None or best_pos_grade < min_grade:
            continue
        if best_pos_grade - bump_grade < min_delta:
            continue

        # Garde la plus pentue, à pente similaire préfère plus longue
        if best is None or best_pos_grade > best['grade'] + 0.3:
            best = {'start_km': dists[best_pos_i]/1000,
                    'length_m': best_pos_length,
                    'grade': best_pos_grade}
        elif abs(best_pos_grade - best['grade']) < 0.3 and best_pos_length > best['length_m']:
            best = {'start_km': dists[best_pos_i]/1000,
                    'length_m': best_pos_length,
                    'grade': best_pos_grade}

    if best:
        return f"{best['length_m']:.0f}m à {best['grade']:.1f}% (km {best['start_km']:.1f})"
    return "Non"


def detect_bumps(coords, elevations,
                 mass_kg=77, CdA=0.30, Crr=0.004):
    """Détecte les bosses avec algo sliding-window grade detection.

    Algorithme :
    1. Resample à 25m
    2. Smooth très léger (50m) juste contre bruit GPX
    3. Calcule pente locale glissante sur 100m en chaque point
    4. Trouve les segments où pente >= seuil au démarrage, étend tant que pente >= seuil loose (1.5%)
    5. S'arrête au premier sommet local (drop > 2m)
    6. Filtre par longueur >= 250m ET pente moyenne >= 3%
    """
    if len(coords) < 4 or len(elevations) != len(coords):
        return []

    dists, elevs = _resample_elevation(coords, elevations, step_m=25)
    if len(dists) < 10:
        return []

    # Smoothing très léger (juste contre bruit GPX), pas l'écrasement à 200m
    smooth = _smooth(elevs, 2)
    n = len(smooth)
    step = 25

    # Pente locale glissante sur 100m
    window_samples = max(2, int(100 / step))
    local_grade = []
    for i in range(n):
        j_lo = max(0, i - window_samples//2)
        j_hi = min(n-1, i + window_samples//2)
        if dists[j_hi] - dists[j_lo] < 1:
            local_grade.append(0)
        else:
            g = (smooth[j_hi] - smooth[j_lo]) / (dists[j_hi] - dists[j_lo]) * 100
            local_grade.append(g)

    bumps = []
    i = 0
    while i < n - 1:
        # Cherche début de bosse : pente locale >= min_grade_pct
        while i < n - 1 and local_grade[i] < 3.0:
            i += 1
        if i >= n - 1:
            break

        start_i = i
        peak_i = i
        peak_val = smooth[i]
        j = i + 1
        max_drop_tolerance = 2.0  # m de descente avant d'arrêter la bosse

        while j < n:
            if smooth[j] >= peak_val:
                peak_val = smooth[j]
                peak_i = j
                j += 1
            else:
                drop = peak_val - smooth[j]
                # Arrêt : descente significative OU pente locale très négative
                if drop > max_drop_tolerance or local_grade[j] < -1.0:
                    break
                j += 1

        length = dists[peak_i] - dists[start_i]
        gain = smooth[peak_i] - smooth[start_i]

        if length >= 250 and gain > 0:
            grade = gain / length * 100

            # Double seuil Wahoo Summit Freeride :
            # - ≥ 400m à ≥ 3% (bosses longues classiques)
            # - OU ≥ 250m à ≥ 7% (bosses courtes punchy)
            # - Interpolation linéaire entre les deux pour les longueurs intermédiaires
            if length >= 400:
                min_grade_required = 3.0
            else:
                # Interpolation : à 250m → 7%, à 400m → 3%
                min_grade_required = 7.0 - 4.0 * (length - 250) / 150

            if grade >= min_grade_required:
                # Pente max sur 100m
                max_g = 0
                for k in range(start_i, peak_i - window_samples + 1):
                    seg_g = (smooth[k + window_samples] - smooth[k]) / 100 * 100
                    if seg_g > max_g:
                        max_g = seg_g

                # Section pointue (nouvelle fonction avec critères stricts)
                section_pointue = _find_pointu_section(smooth, dists, start_i, peak_i, grade)

                # Analyse 2,5km après sommet
                idx_after = min(peak_i + int(2500 / 25), n - 1)
                if idx_after > peak_i:
                    diff = smooth[idx_after] - smooth[peak_i]
                    delta_d = dists[idx_after] - dists[peak_i]
                    g_after = diff / delta_d * 100 if delta_d > 0 else 0
                    if g_after < -1.0:
                        after = f"Descente, {diff:+.0f}m sur 2,5km"
                    elif g_after > 1.0:
                        after = "Montée continue"
                    else:
                        after = "Plat"
                else:
                    after = "Fin du parcours"

                # Simulations physiques
                v250 = _solve_climb_speed(250, grade, mass_kg, CdA, Crr)
                v300 = _solve_climb_speed(300, grade, mass_kg, CdA, Crr)
                v350 = _solve_climb_speed(350, grade, mass_kg, CdA, Crr)



                # FIETS Index (référence académique cyclisme)
                # base = H² / (D × 10), bonus altitude si sommet > 1000m
                fiets = gain**2 / (length * 10)
                if smooth[peak_i] > 1000:
                    fiets += (smooth[peak_i] - 1000) / 1000

                # Catégorie FIETS
                if fiets >= 6.5:
                    fiets_cat = "HC"
                elif fiets >= 5.0:
                    fiets_cat = "Cat 1"
                elif fiets >= 3.5:
                    fiets_cat = "Cat 2"
                elif fiets >= 2.0:
                    fiets_cat = "Cat 3"
                elif fiets >= 0.5:
                    fiets_cat = "Cat 4"
                elif fiets >= 0.25:
                    fiets_cat = "Cat 5"
                else:
                    fiets_cat = "—"

                bumps.append({
                    "num": len(bumps) + 1,
                    "km_start": dists[start_i] / 1000,
                    "length_m": int(length),
                    "grade_avg": round(grade, 1),
                    "grade_max_100m": round(max_g, 1),
                    "section_pointue": section_pointue,
                    "after_summit": after,
                    "time_250w": _format_time(length / v250),
                    "time_300w": _format_time(length / v300),
                    "time_350w": _format_time(length / v350),
                    "d_plus_m": int(gain),
                    "alt_summit_m": int(smooth[peak_i]),
                    "fiets": round(fiets, 2),
                    "fiets_cat": fiets_cat,
                })

        i = peak_i + 1

    return bumps


# ============================================================================
# ALTITUDES
# ============================================================================

def get_route_elevations(route_result):
    """Estime un profil d'altitude pour un itinéraire calculé.

    Le graphe contient le D+ par arête mais pas l'altitude absolue.
    Pour V1, on répartit le D+ total proportionnellement à la distance.
    C'est suffisant pour détecter les bosses (profil relatif), mais
    l'altitude absolue est approximative.
    """
    if not route_result:
        return None
    coords = route_result["coords"]
    total_d_plus = route_result.get("total_d_plus", 0)
    total_length = route_result.get("total_length_m", 0)
    if total_d_plus == 0 or total_length == 0:
        return [100.0] * len(coords)

    elevations = [100.0]
    cum_d = 0.0
    for i in range(1, len(coords)):
        seg = haversine_m(coords[i-1], coords[i])
        cum_d += seg
        ratio = cum_d / max(total_length, 1)
        elevations.append(100.0 + ratio * total_d_plus)
    return elevations


def read_gpx_with_elevations(content):
    """Lit un GPX avec altitudes (cas Strava/Garmin/Wahoo)."""
    import xml.etree.ElementTree as ET
    if isinstance(content, bytes):
        content = content.decode('utf-8')
    coords, elevs = [], []
    try:
        root = ET.fromstring(content)
        for elem in root.iter():
            if elem.tag.endswith('trkpt') or elem.tag.endswith('rtept'):
                lat = float(elem.attrib.get('lat', 0))
                lon = float(elem.attrib.get('lon', 0))
                if not (lat and lon):
                    continue
                coords.append((lat, lon))
                ele = 0.0
                for child in elem:
                    if child.tag.endswith('ele'):
                        try:
                            ele = float(child.text)
                        except (ValueError, TypeError):
                            ele = 0.0
                        break
                elevs.append(ele)
    except Exception as e:
        print(f"Erreur GPX : {e}")
    return coords, elevs

# ============================================================================
# EXPORT EXCEL DES BOSSES
# ============================================================================

def export_bumps_to_excel(bumps, route_name="Itinéraire",
                          mass_total_kg=77, mass_bike_kg=7):

    wb = Workbook()
    ws = wb.active
    ws.title = "Bosses"[:31]

    # Couleurs
    C_HEADER = "FF1A3A5C"
    C_SUB1 = "FF2A609B"
    C_SUB2 = "FF2E86AB"
    C_BAND_LIGHT = "FFFFFFFF"
    C_BAND_ALT = "FFEBF2FA"
    C_POINTU = "FFFFF3CD"
    C_MONTEE = "FFF8D7DA"
    C_PLAT = "FFFFF9C4"
    C_DESCENTE = "FFD4EDDA"
    C_T250 = "FFDBEAFE"
    C_T300 = "FFD1FAE5"
    C_T350 = "FFFCE7F3"

    font_title = Font(color="FFFFFFFF", bold=True, size=14, name="Arial")
    font_sub = Font(color="FFFFFFFF", italic=True, name="Arial")
    font_param = Font(color="FFFFFFFF", name="Arial")
    font_header = Font(color="FFFFFFFF", bold=True, name="Arial")
    font_body = Font(name="Arial")

    align_left = Alignment(horizontal="left", vertical="center")
    align_center = Alignment(horizontal="center", vertical="center", wrap_text=True)

    # Titre
    ws.merge_cells("A1:N1")
    c = ws["A1"]
    c.value = f" ANALYSE DES BOSSES — {route_name}"
    c.fill = PatternFill("solid", fgColor=C_HEADER)
    c.font = font_title
    c.alignment = align_left
    ws.row_dimensions[1].height = 28

    # Sous-titre 1
    ws.merge_cells("A2:N2")
    c = ws["A2"]
    c.value = "Détection des bosses supérieures à 250m et supérieures à 3%"
    c.fill = PatternFill("solid", fgColor=C_SUB1)
    c.font = font_sub
    c.alignment = align_left

    # Sous-titre 2 (paramètres)
    ws.merge_cells("A3:N3")
    c = ws["A3"]
    rider_kg = mass_total_kg - mass_bike_kg
    c.value = (f"Cycliste : {rider_kg} kg  |  Matériel : {mass_bike_kg} kg  |  "
               f"Masse totale : {mass_total_kg} kg  |  Modèle aéro : CdA = 0,30 m²  |  Crr = 0,004")
    c.fill = PatternFill("solid", fgColor=C_SUB2)
    c.font = font_param
    c.alignment = align_left

    # Headers (ligne 5)
    headers = ["N°", "Km début", "Longueur", "Pente moy", "Pente max sur 100m",
               "Section pointue > 100m", "Après le sommet, analyse sur 2,5km",
               "Temps\n250 W", "Temps\n300 W", "Temps\n350 W",
               "D+\n(m)", "Alt. sommet\n(m)", "FIETS", "Cat."]
    for col_idx, h in enumerate(headers, start=1):
        c = ws.cell(row=5, column=col_idx)
        c.value = h
        c.fill = PatternFill("solid", fgColor=C_HEADER)
        c.font = font_header
        c.alignment = align_center
    ws.row_dimensions[5].height = 35

    # Largeurs (12 colonnes initiales + FIETS + Cat.)
    widths = [5, 10, 10, 10, 12, 26, 32, 10, 10, 10, 8, 10, 8, 8]
    for i, w in enumerate(widths, start=1):
        # Pour colonnes au-delà de Z, openpyxl utilise AA, AB...
        if i <= 26:
            col_letter = chr(64+i)
        else:
            col_letter = chr(64 + ((i-1)//26)) + chr(64 + ((i-1) % 26) + 1)
        ws.column_dimensions[col_letter].width = w

    # Données
    for row_idx, b in enumerate(bumps, start=6):
        is_alt = (row_idx % 2 == 0)
        band = C_BAND_ALT if is_alt else C_BAND_LIGHT

        values = [
            b["num"],
            f"{b['km_start']:.1f}".replace(".", ","),
            b["length_m"],
            b["grade_avg"],
            b["grade_max_100m"],
            b["section_pointue"],
            b["after_summit"],
            b["time_250w"],
            b["time_300w"],
            b["time_350w"],
            b["d_plus_m"],
            b["alt_summit_m"],
            b.get("fiets", 0),
            b.get("fiets_cat", "—"),
        ]

        for col_idx, val in enumerate(values, start=1):
            c = ws.cell(row=row_idx, column=col_idx)
            c.value = val
            c.font = font_body
            c.alignment = Alignment(horizontal="center", vertical="center")

            # Couleurs conditionnelles
            if col_idx == 6:
                c.fill = PatternFill("solid", fgColor=C_POINTU if val != "Non" else band)
            elif col_idx == 7:
                vs = str(val)
                if "Descente" in vs:
                    fill = C_DESCENTE
                elif "Montée" in vs:
                    fill = C_MONTEE
                elif "Plat" in vs:
                    fill = C_PLAT
                else:
                    fill = band
                c.fill = PatternFill("solid", fgColor=fill)
            elif col_idx == 8:
                c.fill = PatternFill("solid", fgColor=C_T250)
            elif col_idx == 9:
                c.fill = PatternFill("solid", fgColor=C_T300)
            elif col_idx == 10:
                c.fill = PatternFill("solid", fgColor=C_T350)
            else:
                c.fill = PatternFill("solid", fgColor=band)

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.read()


# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar():
    """Affiche la sidebar et retourne les paramètres de config."""
    with st.sidebar:
        st.header("⚙️ Configuration")

        profile = st.selectbox(
            "Profil",
            options=["club_road", "solo_casual"],
            format_func=lambda x: "🚴 Club road" if x == "club_road" else "🚲 Solo casual",
        )

        alpha = st.slider(
            "Préférence plaisir vs distance",
            min_value=0.0, max_value=2.5, value=1.0, step=0.1,
            help="0 = plus court, 1 = équilibré, 2+ = priorité plaisir",
        )

        st.divider()
        st.subheader("🏔️ Mode grimpeur")
        use_climber = st.checkbox("Activer budget D+", value=False)
        d_plus_target = None
        if use_climber:
            d_plus_target = st.number_input(
                "D+ cible (m)", value=800, min_value=0, max_value=5000, step=100
            )

        st.divider()
        st.subheader("🗺️ Affichage")
        show_club = st.checkbox("Afficher toutes les traces club", value=False)
        show_uploaded = st.checkbox("Afficher trace uploadée", value=True)

        st.divider()
        if st.button("🔄 Réinitialiser waypoints"):
            for k in ["waypoints", "route_result", "bumps_route"]:
                st.session_state[k] = [] if k == "waypoints" else None
            st.rerun()

        st.divider()
        st.subheader("🔍 Recherche d'adresse")
        address = st.text_input("Adresse ou lieu",
            placeholder="Versailles, Bastille, Chevreuse...")
        if st.button("📍 Ajouter ce point", disabled=not address):
            result = geocode_address(address)
            if result:
                lat, lon, display = result
                st.session_state.waypoints.append((lat, lon))
                st.success(f"✅ {display[:60]}")
                st.rerun()
            else:
                st.error("Adresse non trouvée en IdF")

    return {
        "profile": profile,
        "alpha": alpha,
        "use_climber": use_climber,
        "d_plus_target": d_plus_target,
        "show_club": show_club,
        "show_uploaded": show_uploaded,
    }


# ============================================================================
# CARTE
# ============================================================================

def build_map(config, club_traces):
    """Construit l'objet folium.Map avec tous les calques actifs."""
    waypoints = st.session_state.waypoints
    center = [48.80, 2.30]
    if waypoints:
        center = [np.mean([w[0] for w in waypoints]),
                  np.mean([w[1] for w in waypoints])]

    m = folium.Map(location=center, zoom_start=10, tiles="cartodbpositron")

    # Traces club (~149 simplifiées)
    if config["show_club"]:
        simplified = cached_simplified_traces(club_traces)
        fg = folium.FeatureGroup(name=f"Traces club ({len(simplified)})")
        for t in simplified:
            folium.PolyLine(
                t["coords"], color="purple",
                weight=1.5, opacity=0.4, smooth_factor=2,
                tooltip=t["name"][:40],
            ).add_to(fg)
        fg.add_to(m)

    # Trace uploadée
    if config["show_uploaded"] and st.session_state.uploaded_trace:
        folium.PolyLine(
            st.session_state.uploaded_trace,
            color="orange", weight=4, opacity=0.7,
            tooltip="Trace uploadée",
        ).add_to(m)

    # Itinéraire calculé
    if st.session_state.route_result:
        r = st.session_state.route_result
        folium.PolyLine(
            r["coords"], color="red", weight=4, opacity=0.9,
            tooltip=f"Itinéraire ({r['total_length_m']/1000:.1f} km)",
        ).add_to(m)

    # Waypoints
    for i, (lat, lon) in enumerate(waypoints):
        if i == 0:
            icon, label = folium.Icon(color="green", icon="play"), "🚩 Départ"
        elif i == len(waypoints) - 1 and len(waypoints) > 1:
            icon, label = folium.Icon(color="red", icon="stop"), "🏁 Arrivée"
        else:
            icon, label = folium.Icon(color="blue", icon="info-sign"), f"Waypoint {i}"
        folium.Marker([lat, lon], icon=icon, tooltip=label).add_to(m)

    return m


# ============================================================================
# WAYPOINTS + ROUTING
# ============================================================================

def render_waypoints_panel():
    """Affiche la liste des waypoints placés."""
    st.subheader("📍 Waypoints")
    waypoints = st.session_state.waypoints
    if not waypoints:
        st.write("_Clique sur la carte pour placer un point._")
        return

    for i, (lat, lon) in enumerate(waypoints):
        if i == 0:
            emoji = "🚩"
        elif i == len(waypoints) - 1 and len(waypoints) > 1:
            emoji = "🏁"
        else:
            emoji = f"{i}️⃣"
        st.text(f"{emoji} {lat:.4f}, {lon:.4f}")

    # Détection boucle
    if len(waypoints) >= 3:
        d = ((waypoints[0][0] - waypoints[-1][0])**2 +
             (waypoints[0][1] - waypoints[-1][1])**2) ** 0.5
        if d < 0.005:
            st.success("🔁 Boucle détectée")


def trigger_routing(G, config, club_traces):
    """Bouton calculer + lancement du routing + détection bosses."""
    waypoints = st.session_state.waypoints
    can_route = len(waypoints) >= 2

    if st.button("🚀 Calculer l'itinéraire", type="primary", disabled=not can_route):
        with st.spinner("Calcul..."):
            if config["use_climber"]:
                best = None
                best_diff = float('inf')
                for a in [0.5, 1.0, 1.5, 2.0]:
                    r = route_waypoints(G, waypoints, profile=config["profile"], alpha=a)
                    if r:
                        diff = abs(r["total_d_plus"] - config["d_plus_target"])
                        if diff < best_diff:
                            best = r
                            best_diff = diff
                result = best
            else:
                result = route_waypoints(G, waypoints, profile=config["profile"], alpha=config["alpha"])

        if result:
            st.session_state.route_result = result
            st.session_state.comparison = similarity_to_club(result["coords"], club_traces, top_k=5)
            # Bosses sur itinéraire calculé
            elevs = get_route_elevations(result)
            if elevs:
                st.session_state.bumps_route = detect_bumps(result["coords"], elevs)
            st.rerun()
        else:
            st.error("Pas de chemin trouvé entre ces points")


# ============================================================================
# STATS + SIMILARITÉ
# ============================================================================

def render_route_stats(config):
    """Métriques de l'itinéraire calculé + download GPX."""
    if not st.session_state.route_result:
        return
    r = st.session_state.route_result
    st.subheader("📊 Itinéraire calculé")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Distance", f"{r['total_length_m']/1000:.1f} km")
        st.metric("Score plaisir", f"{r['mean_score']:.2f}")
    with c2:
        st.metric("D+", f"{r['total_d_plus']:.0f} m")
        st.metric("Waypoints", r.get("n_waypoints", 2))

    gpx_content = export_gpx_string(
        r["coords"],
        name=f"Itineraire_{config['profile']}_a{config['alpha']}",
        description=f"{r['total_length_m']/1000:.1f}km, D+{r['total_d_plus']:.0f}m"
    )
    st.download_button(
        "📥 Télécharger GPX",
        data=gpx_content,
        file_name=f"itineraire_{config['profile']}.gpx",
        mime="application/gpx+xml",
    )


def render_similarity():
    """Affiche similarité de l'itinéraire calculé avec les traces club."""
    if not st.session_state.comparison:
        return
    c = st.session_state.comparison
    st.subheader("🎯 Similarité club")
    st.metric("Coverage", f"{c['coverage']*100:.0f}%")
    st.metric("F1 (top 5)", f"{c['global_f1']*100:.0f}%")

    if c["best_matches"]:
        with st.expander("Top 5 traces similaires"):
            for m in c["best_matches"]:
                st.text(f"{m['name'][:30]} — F1 {m['f1']*100:.0f}%")



# ============================================================================
# UPLOAD GPX
# ============================================================================

def render_gpx_upload(club_traces):
    """Upload GPX, comparaison F1 club, détection bosses."""
    st.subheader("📤 Upload trace")
    uploaded = st.file_uploader("Drop un GPX", type=["gpx"])

    if uploaded:
        content = uploaded.read()
        coords, elevs = read_gpx_with_elevations(content)

        if coords:
            st.session_state.uploaded_trace = coords
            st.session_state.uploaded_elevs = elevs
            st.session_state.uploaded_sim = similarity_to_club(coords, club_traces, top_k=5)
            # Détection bosses avec les vraies altitudes du GPX
            if elevs and max(elevs) > 0:
                st.session_state.bumps_uploaded = detect_bumps(coords, elevs)
            else:
                st.session_state.bumps_uploaded = None
            st.success(f"✅ {len(coords)} points")
        else:
            st.error("Aucun point dans le GPX")
            for k in ["uploaded_trace", "uploaded_elevs", "uploaded_sim", "bumps_uploaded"]:
                st.session_state[k] = None

    # Similarité GPX avec TOUTES les traces club
    if st.session_state.uploaded_sim:
        sim = st.session_state.uploaded_sim
        st.markdown("**📊 GPX vs club**")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Coverage", f"{sim['coverage']*100:.0f}%")
        with col_b:
            st.metric("F1 (top 5)", f"{sim['global_f1']*100:.0f}%")

        if sim["best_matches"]:
            with st.expander("Top 5 traces club similaires au GPX"):
                for m in sim["best_matches"]:
                    st.text(f"{m['name'][:35]} — F1 {m['f1']*100:.0f}%")



# ============================================================================
# VISUALISATION DES BOSSES
# ============================================================================

def _style_bump_row(row):
    """Couleurs ligne par ligne dans le DataFrame Streamlit."""
    styles = [""] * len(row)
    cols = list(row.index)

    if "Section pointue" in cols:
        idx = cols.index("Section pointue")
        if row["Section pointue"] != "Non":
            styles[idx] = "background-color: #FFF3CD"

    if "Après sommet" in cols:
        idx = cols.index("Après sommet")
        val = str(row["Après sommet"])
        if "Descente" in val:
            styles[idx] = "background-color: #D4EDDA"
        elif "Montée" in val:
            styles[idx] = "background-color: #F8D7DA"
        elif "Plat" in val:
            styles[idx] = "background-color: #FFF9C4"

    # Colonnes temps
    for col_name, color in [("Temps 250W", "#DBEAFE"),
                             ("Temps 300W", "#D1FAE5"),
                             ("Temps 350W", "#FCE7F3")]:
        if col_name in cols:
            idx = cols.index(col_name)
            if not styles[idx]:
                styles[idx] = f"background-color: {color}"

    return styles


def _show_bumps_table(bumps, name, fname):
    """Affiche tableau bosses + bouton Excel."""
    if not bumps:
        st.info("Aucune bosse détectée (seuils : >250m et >3%)")
        return

    df = pd.DataFrame([{
        "N°": b["num"],
        "Km début": f"{b['km_start']:.1f}".replace(".", ","),
        "Longueur (m)": b["length_m"],
        "Pente moy (%)": b["grade_avg"],
        "Pente max 100m": b["grade_max_100m"],
        "Section pointue": b["section_pointue"],
        "Après sommet": b["after_summit"],
        "Temps 250W": b["time_250w"],
        "Temps 300W": b["time_300w"],
        "Temps 350W": b["time_350w"],
        "D+ (m)": b["d_plus_m"],
        "Alt. sommet": b["alt_summit_m"],
        "FIETS": b.get("fiets", "—"),
        "Cat.": b.get("fiets_cat", "—"),
    } for b in bumps])

    styled = df.style.apply(_style_bump_row, axis=1)
    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Pente moy (%)": st.column_config.NumberColumn(format="%.1f"),
            "Pente max 100m": st.column_config.NumberColumn(format="%.1f"),
        },
    )

    # Bouton Excel stylé
    xlsx_bytes = export_bumps_to_excel(bumps, route_name=name)
    st.download_button(
        f"📥 Télécharger Excel ({name})",
        data=xlsx_bytes,
        file_name=f"bosses_{fname}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"dl_{fname}",
    )

    # Stats
    total_dp = sum(b["d_plus_m"] for b in bumps)
    avg_g = np.mean([b['grade_avg'] for b in bumps])
    st.caption(f"📊 {len(bumps)} bosses · {total_dp}m D+ cumulé · "
               f"pente moyenne {avg_g:.1f}%")


def render_bumps_section():
    """Section bosses sous la carte. Tabs si itinéraire + GPX."""
    bumps_route = st.session_state.get("bumps_route")
    bumps_uploaded = st.session_state.get("bumps_uploaded")

    if not bumps_route and not bumps_uploaded:
        return

    st.divider()
    st.subheader("⛰️ Analyse des bosses")
    st.caption("Détection : longueur ≥ 250m ET pente moyenne ≥ 3% — "
               "Simulation : 77 kg, CdA 0.30 m², Crr 0.004")

    to_show = []
    if bumps_route:
        to_show.append(("Itinéraire calculé", bumps_route, "itineraire"))
    if bumps_uploaded:
        to_show.append(("GPX uploadé", bumps_uploaded, "gpx_upload"))

    if len(to_show) == 1:
        name, bumps, fname = to_show[0]
        _show_bumps_table(bumps, name, fname)
    else:
        tabs = st.tabs([t[0] for t in to_show])
        for tab, (name, bumps, fname) in zip(tabs, to_show):
            with tab:
                _show_bumps_table(bumps, name, fname)



# ============================================================================
# MAIN
# ============================================================================

def main():
    init_session_state()

    G = cached_graph()
    club_traces = cached_traces()

    config = render_sidebar()

    col_map, col_info = st.columns([3, 1])

    with col_map:
        st.subheader("Carte")

        n_wp = len(st.session_state.waypoints)
        if n_wp == 0:
            st.info("👇 Clique sur la carte pour placer ton **départ**")
        elif n_wp == 1:
            st.info("👇 Clique pour placer ton **arrivée** ou un waypoint")
        else:
            st.info(f"✅ {n_wp} points · ajoute d'autres waypoints ou clique **Calculer**")

        m = build_map(config, club_traces)
        map_data = st_folium(m, height=600, width=None, returned_objects=["last_clicked"])

        if map_data and map_data.get("last_clicked"):
            clicked = map_data["last_clicked"]
            new_wp = (clicked["lat"], clicked["lng"])
            if not st.session_state.waypoints or st.session_state.waypoints[-1] != new_wp:
                st.session_state.waypoints.append(new_wp)
                st.rerun()

    with col_info:
        render_waypoints_panel()
        st.divider()
        trigger_routing(G, config, club_traces)
        st.divider()
        render_route_stats(config)
        st.divider()
        render_similarity()
        st.divider()
        render_gpx_upload(club_traces)

    # Section bosses en pleine largeur sous la carte
    render_bumps_section()

    # Footer
    st.divider()
    with st.expander("ℹ️ Aide"):
        st.markdown("""
        """)


if __name__ == "__main__":
    main()

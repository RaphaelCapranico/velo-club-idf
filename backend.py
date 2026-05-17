"""
backend.py — Fonctions de routing et utilitaires pour Streamlit.
Généré par Notebook 11. Ne pas modifier directement, regénérer depuis le notebook.
"""

import os
import pickle
from pathlib import Path
from math import radians, sin, cos, atan2, sqrt
from datetime import datetime
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import networkx as nx
from sqlalchemy import create_engine, text
from sklearn.neighbors import KDTree
from shapely import wkt


# =============================================================================
# CONFIG
# =============================================================================

DB_CONFIG = {
    "user":     os.environ.get("VELO_DB_USER", "postgres"),
    "password": os.environ.get("VELO_DB_PASSWORD", "4421"),
    "host":     os.environ.get("VELO_DB_HOST", "localhost"),
    "port":     int(os.environ.get("VELO_DB_PORT", 5432)),
    "database": os.environ.get("VELO_DB_NAME", "velo_club"),
}

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)
GRAPH_CACHE = CACHE_DIR / "graph.pkl"
TRACES_CACHE = CACHE_DIR / "club_traces.pkl"


def get_engine():
    url = (f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
           f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
    return create_engine(url, pool_pre_ping=True)


# =============================================================================
# GRAPHE
# =============================================================================

def _build_kdtree(G):
    """(Re)construit le KDTree pour snap_to_node. Appelé après chaque chargement.
    Ne pas le mettre dans le pickle pour rester compatible entre versions sklearn."""
    nodes = list(G.nodes(data=True))
    coords = np.array([(d['lat'], d['lon']) for _, d in nodes])
    coords_m = _latlon_to_meters_arr(coords)
    G.graph['kdtree'] = KDTree(coords_m)
    G.graph['node_ids'] = [n for n, _ in nodes]
    return G


def load_graph(use_cache=True):
    """Charge le graphe networkx. Cache pickle pour rapidité.
    Le KDTree est reconstruit après chaque chargement (pas dans le pickle)."""
    if use_cache and GRAPH_CACHE.exists():
        with open(GRAPH_CACHE, 'rb') as f:
            G = pickle.load(f)
        # KDTree absent ou incompatible : on le reconstruit
        if 'kdtree' not in G.graph or not _kdtree_works(G):
            G = _build_kdtree(G)
        return G

    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT
                e.edge_id, e.length_m, e.highway, e.d_plus_m,
                ROUND(ST_X(ST_StartPoint(e.geom))::numeric, 6) AS u_lon,
                ROUND(ST_Y(ST_StartPoint(e.geom))::numeric, 6) AS u_lat,
                ROUND(ST_X(ST_EndPoint(e.geom))::numeric, 6) AS v_lon,
                ROUND(ST_Y(ST_EndPoint(e.geom))::numeric, 6) AS v_lat,
                COALESCE(es.cost_factor_v2_club, es.cost_factor_club, 1.0) AS cost_factor_club,
                COALESCE(es.cost_factor_v2_solo, es.cost_factor_solo, 1.0) AS cost_factor_solo,
                COALESCE(es.score_final_club, 0) AS score_final_club,
                COALESCE(es.score_final_solo, 0) AS score_final_solo
            FROM osm_edges e
            LEFT JOIN edge_scores es ON es.edge_id = e.edge_id
            WHERE e.is_routable = TRUE AND e.geom IS NOT NULL AND e.length_m > 0;
        """), conn)

    df["u_id"] = df["u_lat"].astype(str) + "_" + df["u_lon"].astype(str)
    df["v_id"] = df["v_lat"].astype(str) + "_" + df["v_lon"].astype(str)

    G = nx.MultiDiGraph()
    for _, r in df.iterrows():
        G.add_node(r.u_id, lat=float(r.u_lat), lon=float(r.u_lon))
        G.add_node(r.v_id, lat=float(r.v_lat), lon=float(r.v_lon))
        attrs = {
            "edge_id": int(r.edge_id),
            "length_m": float(r.length_m),
            "highway": r.highway,
            "d_plus_m": float(r.d_plus_m or 0),
            "cost_factor_club": float(r.cost_factor_club),
            "cost_factor_solo": float(r.cost_factor_solo),
            "score_final_club": float(r.score_final_club),
            "score_final_solo": float(r.score_final_solo),
        }
        G.add_edge(r.u_id, r.v_id, **attrs)
        G.add_edge(r.v_id, r.u_id, **attrs)

    largest_cc = max(nx.weakly_connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()

    # On sauvegarde le pickle SANS le KDTree (sera reconstruit au chargement)
    with open(GRAPH_CACHE, 'wb') as f:
        pickle.dump(G, f)

    # Ajoute le KDTree pour utilisation immédiate
    G = _build_kdtree(G)
    return G


def _kdtree_works(G):
    """Teste si le KDTree existant est utilisable (compatible sklearn version)."""
    try:
        kdt = G.graph.get('kdtree')
        if kdt is None:
            return False
        # Test d'un query simple
        kdt.query(np.array([[0, 0]]), k=1)
        return True
    except Exception:
        return False


def reset_graph_cache():
    """Force le rechargement du graphe au prochain appel."""
    if GRAPH_CACHE.exists():
        GRAPH_CACHE.unlink()


# =============================================================================
# ROUTING
# =============================================================================

def _latlon_to_meters_arr(coords_arr):
    LAT_REF = 48.8
    lat_rad = np.radians(LAT_REF)
    x = coords_arr[:, 1] * 111320 * np.cos(lat_rad)
    y = coords_arr[:, 0] * 111320
    return np.column_stack([x, y])


def snap_to_node(G, lat, lon):
    """Trouve le nœud le plus proche via KDTree."""
    if 'kdtree' not in G.graph:
        # Fallback : recherche linéaire
        best_id, best_d = None, float('inf')
        for nid, d in G.nodes(data=True):
            dlat = (d['lat'] - lat) * 111320
            dlon = (d['lon'] - lon) * 111320 * np.cos(np.radians(lat))
            dist = dlat*dlat + dlon*dlon
            if dist < best_d:
                best_d = dist
                best_id = nid
        return best_id

    pt = _latlon_to_meters_arr(np.array([[lat, lon]]))
    _, idx = G.graph['kdtree'].query(pt, k=1)
    return G.graph['node_ids'][int(idx[0][0])]


def edge_cost(attrs, key, alpha):
    L = attrs["length_m"]
    cf = attrs[key]
    return L * (cf ** alpha)


def route(G, start_latlon, end_latlon, profile="club_road", alpha=1.0):
    """Dijkstra A → B."""
    u = snap_to_node(G, *start_latlon)
    v = snap_to_node(G, *end_latlon)
    cost_key = f"cost_factor_{profile.replace('_road', '').replace('_casual', '')}"

    def cost_fn(a, b, d):
        if isinstance(d, dict) and any(isinstance(x, dict) for x in d.values()):
            return min(edge_cost(attrs, cost_key, alpha) for attrs in d.values())
        return edge_cost(d, cost_key, alpha)

    try:
        path = nx.dijkstra_path(G, u, v, weight=cost_fn)
    except nx.NetworkXNoPath:
        return None

    total_L = 0
    total_d_plus = 0
    weighted_score = 0
    coords = [(G.nodes[path[0]]['lat'], G.nodes[path[0]]['lon'])]
    score_key = f"score_final_{profile.replace('_road', '').replace('_casual', '')}"
    edge_ids = []

    for i in range(len(path) - 1):
        a, b = path[i], path[i+1]
        cands = G[a][b]
        best = min(cands.keys(), key=lambda k: edge_cost(cands[k], cost_key, alpha))
        attrs = cands[best]
        total_L += attrs["length_m"]
        total_d_plus += attrs.get("d_plus_m", 0)
        weighted_score += attrs["length_m"] * attrs[score_key]
        coords.append((G.nodes[b]['lat'], G.nodes[b]['lon']))
        edge_ids.append(attrs["edge_id"])

    return {
        "coords": coords,
        "total_length_m": total_L,
        "total_d_plus": total_d_plus,
        "mean_score": weighted_score / max(total_L, 1),
        "profile": profile,
        "alpha": alpha,
        "edge_ids": edge_ids,
    }


def route_waypoints(G, waypoints, profile="club_road", alpha=1.0):
    """Routing par séquence de waypoints. waypoints = [(lat, lon), ...].
    Si premier = dernier, on a une boucle."""
    if len(waypoints) < 2:
        return None

    all_coords = []
    total_L = 0
    total_d_plus = 0
    weighted_score = 0
    all_edge_ids = []

    for i in range(len(waypoints) - 1):
        leg = route(G, waypoints[i], waypoints[i+1], profile, alpha)
        if leg is None:
            return None
        if i == 0:
            all_coords.extend(leg["coords"])
        else:
            all_coords.extend(leg["coords"][1:])
        total_L += leg["total_length_m"]
        total_d_plus += leg["total_d_plus"]
        weighted_score += leg["mean_score"] * leg["total_length_m"]
        all_edge_ids.extend(leg["edge_ids"])

    return {
        "coords": all_coords,
        "total_length_m": total_L,
        "total_d_plus": total_d_plus,
        "mean_score": weighted_score / max(total_L, 1),
        "profile": profile,
        "alpha": alpha,
        "edge_ids": all_edge_ids,
        "n_waypoints": len(waypoints),
    }


# =============================================================================
# TRACES CLUB
# =============================================================================

def load_club_traces(use_cache=True):
    """Charge toutes les traces club avec cache pickle."""
    if use_cache and TRACES_CACHE.exists():
        with open(TRACES_CACHE, 'rb') as f:
            return pickle.load(f)

    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT trace_id, name, ST_AsText(geom) AS wkt_geom
            FROM traces WHERE geom IS NOT NULL;
        """), conn)

    traces = []
    for _, r in df.iterrows():
        try:
            g = wkt.loads(r["wkt_geom"])
            if hasattr(g, "geoms"):
                coords = []
                for line in g.geoms:
                    coords.extend([(y, x) for x, y in line.coords])
            else:
                coords = [(y, x) for x, y in g.coords]
            if len(coords) >= 2:
                traces.append({
                    "trace_id": int(r["trace_id"]),
                    "name": r["name"] or f"Trace {r['trace_id']}",
                    "coords": coords,
                })
        except Exception:
            continue

    with open(TRACES_CACHE, 'wb') as f:
        pickle.dump(traces, f)
    return traces


# =============================================================================
# SIMILARITÉ F1 GÉO
# =============================================================================

def f1_geo(coords_a, coords_b, threshold_m=50):
    """F1 géographique entre 2 traces."""
    if not coords_a or not coords_b:
        return {"precision": 0, "recall": 0, "f1": 0}

    pts_a = _latlon_to_meters_arr(np.array(coords_a))
    pts_b = _latlon_to_meters_arr(np.array(coords_b))

    tree_b = KDTree(pts_b)
    dists_a, _ = tree_b.query(pts_a, k=1)
    precision = float((dists_a.flatten() < threshold_m).mean())

    tree_a = KDTree(pts_a)
    dists_b, _ = tree_a.query(pts_b, k=1)
    recall = float((dists_b.flatten() < threshold_m).mean())

    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    return {"precision": precision, "recall": recall, "f1": f1}


def similarity_to_club(coords_test, club_traces, threshold_m=50, top_k=5):
    """Similarité d'une trace à l'ensemble des traces club."""
    if not coords_test or not club_traces:
        return {"coverage": 0, "best_matches": [], "global_f1": 0}

    pts_test = _latlon_to_meters_arr(np.array(coords_test))

    # Coverage : % de la trace test à < threshold d'au moins une trace club
    all_club_pts = []
    for t in club_traces:
        all_club_pts.extend(t["coords"])
    pts_club = _latlon_to_meters_arr(np.array(all_club_pts))
    tree_club = KDTree(pts_club)
    dists, _ = tree_club.query(pts_test, k=1)
    coverage = float((dists.flatten() < threshold_m).mean())

    # F1 individuel avec chaque trace
    matches = []
    for t in club_traces:
        r = f1_geo(coords_test, t["coords"], threshold_m)
        if r["f1"] > 0.05:  # Skip les non-pertinentes
            matches.append({
                "trace_id": t["trace_id"],
                "name": t["name"],
                "f1": r["f1"],
                "precision": r["precision"],
                "recall": r["recall"],
            })
    matches.sort(key=lambda x: -x["f1"])

    return {
        "coverage": coverage,
        "best_matches": matches[:top_k],
        "global_f1": float(np.mean([m["f1"] for m in matches[:top_k]])) if matches else 0,
    }


# =============================================================================
# GPX I/O
# =============================================================================

def haversine_m(p1, p2):
    R = 6371000
    phi1, phi2 = radians(p1[0]), radians(p2[0])
    dphi = radians(p2[0] - p1[0])
    dlam = radians(p2[1] - p1[1])
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlam/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1-a))


def smooth_path(coords, min_step_m=30):
    if len(coords) < 3:
        return coords
    smoothed = [coords[0]]
    for pt in coords[1:-1]:
        if haversine_m(smoothed[-1], pt) >= min_step_m:
            smoothed.append(pt)
    smoothed.append(coords[-1])
    return smoothed


def densify_path(coords, target_step_m=25):
    if len(coords) < 2:
        return coords
    dense = [coords[0]]
    for i in range(1, len(coords)):
        p1 = dense[-1]
        p2 = coords[i]
        d = haversine_m(p1, p2)
        if d > target_step_m:
            n_inserts = int(d // target_step_m)
            for j in range(1, n_inserts + 1):
                t = j / (n_inserts + 1)
                interp = (p1[0] + t * (p2[0] - p1[0]),
                          p1[1] + t * (p2[1] - p1[1]))
                dense.append(interp)
        dense.append(p2)
    return dense


def export_gpx_string(coords, name="Itineraire", description=""):
    """Génère le contenu GPX comme chaîne (pour download Streamlit)."""
    coords = smooth_path(coords, 30)
    coords = densify_path(coords, 25)

    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    xml = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<gpx version="1.1" creator="VeloClubIDF" '
           'xmlns="http://www.topografix.com/GPX/1/1">',
           '  <metadata>',
           f'    <name>{name}</name>',
           f'    <desc>{description}</desc>',
           f'    <time>{timestamp}</time>',
           '  </metadata>',
           '  <trk>',
           f'    <name>{name}</name>',
           '    <trkseg>']
    for lat, lon in coords:
        xml.append(f'      <trkpt lat="{lat:.7f}" lon="{lon:.7f}"><ele>0</ele></trkpt>')
    xml.extend(['    </trkseg>', '  </trk>', '</gpx>'])
    return "\n".join(xml)


def read_gpx_string(content):
    """Parse un GPX (string ou bytes) et retourne les coords."""
    if isinstance(content, bytes):
        content = content.decode('utf-8')

    coords = []
    try:
        root = ET.fromstring(content)
        for elem in root.iter():
            if elem.tag.endswith('trkpt') or elem.tag.endswith('rtept'):
                lat = float(elem.attrib.get('lat', 0))
                lon = float(elem.attrib.get('lon', 0))
                if lat and lon:
                    coords.append((lat, lon))
    except Exception as e:
        print(f"Erreur GPX : {e}")
    return coords

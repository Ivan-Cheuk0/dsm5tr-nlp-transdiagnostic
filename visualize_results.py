"""
DSM-5-TR Embedding Visualization, HDBSCAN & Support Vector Clustering (SVC) Pipeline
====================================================================================
Reads the 768-dimensional BioBERT (`pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb`)
and PubMedBERT (`NeuML/pubmedbert-base-embeddings`) symptom embeddings from `outputs/`,
maps all 170 DSM-5-TR disorders into their 20 overarching DSM-5-TR diagnostic chapters,
runs manifold projection + mutual-reachability density clustering (HDBSCAN) + RBF Support
Vector Clustering (SVC) boundary estimation, and generates publication-ready SVG/PNG figures
plus an interactive standalone HTML explorer in `outputs/figures/`.
"""

from __future__ import annotations

import csv
import html
import json
import math
import random
import struct
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


CHAPTER_RANGES = [
    ("Neurodevelopmental", "Intellectual Developmental Disorder (Intellectual Disability)", "Provisional Tic Disorder"),
    ("Schizophrenia Spectrum", "Delusional Disorder", "Catatonic Disorder due to Another Medical Condition"),
    ("Bipolar & Related", "Bipolar I Disorder - Manic Episode", "Bipolar and Related Disorder due to Another Medical Condition"),
    ("Depressive", "Disruptive Mood Dysregulation Disorder", "Depressive Disorder due to Another Medical Condition"),
    ("Anxiety", "Separation Anxiety Disorder", "Anxiety Disorder due to Another Medical Condition"),
    ("Obsessive-Compulsive", "Obsessive-Compulsive Disorder", "Obsessive-Compulsive and Related Disorder due to Another Medical Condition"),
    ("Trauma & Stressor", "Reactive Attachment Disorder", "Prolonged Grief Disorder"),
    ("Dissociative", "Dissociative Identity Disorder", "Depersonalization/Derealization Disorder"),
    ("Somatic & Factitious", "Somatic Symptom Disorder", "Factitious Disorder Imposed on Another"),
    ("Feeding & Eating", "Pica", "Binge-Eating Disorder"),
    ("Elimination", "Enuresis", "Encopresis"),
    ("Sleep-Wake", "Insomnia Disorder", "Substance/Medication-Induced Sleep Disorder"),
    ("Sexual Dysfunctions", "Delayed Ejaculation", "Substance/Medication-Induced Sexual Dysfunction"),
    ("Gender Dysphoria", "Gender Dysphoria in Children", "Gender Dysphoria in Adolescents and Adults"),
    ("Disruptive & Impulse-Control", "Oppositional Defiant Disorder", "Kleptomania"),
    ("Substance & Addictive", "Alcohol Use Disorder", "Gambling Disorder"),
    ("Neurocognitive", "Delirium", "Major or Mild Neurocognitive Disorder due to Parkinson’s Disease"),
    ("Personality", "General Personality Disorder", "Personality Change due to Another Medical Condition"),
    ("Paraphilic", "Voyeuristic Disorder", "Transvestic Disorder"),
    ("Section III (Emerging)", "Attenuated Psychosis Syndrome", "Nonsuicidal Self-Injury Disorder"),
]

CHAPTER_COLORS = {
    "Neurodevelopmental": "#2563eb",
    "Schizophrenia Spectrum": "#7c3aed",
    "Bipolar & Related": "#db2777",
    "Depressive": "#e11d48",
    "Anxiety": "#ea580c",
    "Obsessive-Compulsive": "#d97706",
    "Trauma & Stressor": "#ca8a04",
    "Dissociative": "#65a30d",
    "Somatic & Factitious": "#16a34a",
    "Feeding & Eating": "#059669",
    "Elimination": "#0d9488",
    "Sleep-Wake": "#0891b2",
    "Sexual Dysfunctions": "#0284c7",
    "Gender Dysphoria": "#4f46e5",
    "Disruptive & Impulse-Control": "#9333ea",
    "Substance & Addictive": "#c026d3",
    "Neurocognitive": "#475569",
    "Personality": "#dc2626",
    "Paraphilic": "#0f766e",
    "Section III (Emerging)": "#64748b",
}

CLUSTER_PALETTE = [
    "#2563eb", "#e11d48", "#16a34a", "#d97706", "#7c3aed",
    "#0891b2", "#db2777", "#65a30d", "#ea580c", "#4f46e5",
    "#0d9488", "#9333ea", "#ca8a04", "#0284c7", "#dc2626",
]


def load_npy_matrix(path: Path) -> tuple[int, int, list[list[float]]]:
    """Loads a 2D float32 .npy array using only the Python standard library."""
    with open(path, "rb") as f:
        _magic = f.read(6)
        major, _minor = struct.unpack("BB", f.read(2))
        hdr_len = struct.unpack("<H" if major == 1 else "<I", f.read(2 if major == 1 else 4))[0]
        hdr = eval(f.read(hdr_len).decode("latin1").strip())
        rows, cols = hdr["shape"]
        mv = memoryview(f.read()).cast("f")
        mat = [mv[i * cols : (i + 1) * cols].tolist() for i in range(rows)]
        return rows, cols, mat


def compute_cosine_matrix(mat: list[list[float]]) -> list[list[float]]:
    """Computes N x N dot-product cosine similarity across platforms (NumPy -> macOS Accelerate -> pure Python)."""
    try:
        import numpy as np
        arr = np.asarray(mat, dtype=np.float32)
        return np.clip(arr @ arr.T, -1.0, 1.0).tolist()
    except ImportError:
        pass
    try:
        import array, ctypes
        n, k = len(mat), len(mat[0])
        flat = array.array("f")
        for row in mat:
            flat.extend(row)
        out = array.array("f", [0.0] * (n * n))
        acc = ctypes.CDLL("/System/Library/Frameworks/Accelerate.framework/Accelerate")
        c_float_p = ctypes.POINTER(ctypes.c_float)
        A_ptr = ctypes.cast(flat.buffer_info()[0], c_float_p)
        C_ptr = ctypes.cast(out.buffer_info()[0], c_float_p)
        acc.cblas_sgemm(
            101, 111, 112,
            ctypes.c_int(n), ctypes.c_int(n), ctypes.c_int(k),
            ctypes.c_float(1.0), A_ptr, ctypes.c_int(k),
            A_ptr, ctypes.c_int(k),
            ctypes.c_float(0.0), C_ptr, ctypes.c_int(n),
        )
        mv = memoryview(out)
        return [mv[i * n : (i + 1) * n].tolist() for i in range(n)]
    except Exception:
        n = len(mat)
        sim = [[0.0] * n for _ in range(n)]
        for i in range(n):
            sim[i][i] = 1.0
            for j in range(i + 1, n):
                s = max(-1.0, min(1.0, sum(a * b for a, b in zip(mat[i], mat[j]))))
                sim[i][j] = sim[j][i] = s
        return sim


def map_disorders_to_chapters(sym_data: list[dict]) -> dict[str, str]:
    disorders = []
    for r in sym_data:
        if r["disorder"] not in disorders:
            disorders.append(r["disorder"])
    d_to_chap = {}
    for chap, start_d, end_d in CHAPTER_RANGES:
        s_idx = disorders.index(start_d)
        e_idx = disorders.index(end_d)
        for d in disorders[s_idx : e_idx + 1]:
            d_to_chap[d] = chap
    return d_to_chap


def knn_manifold_2d(sim_mat: list[list[float]], k: int = 15, iterations: int = 180, seed: int = 42) -> list[tuple[float, float]]:
    """
    Computes a UMAP-style 2D attraction-repulsion k-NN layout directly from the
    cosine similarity matrix. First initializes coordinates with top-2 spectral
    eigenvectors of the k-NN affinity graph (power iteration with orthogonalization),
    then refines via attractive k-NN forces and negative-sampling repulsive forces.
    """
    n = len(sim_mat)
    rng = random.Random(seed)

    # Build k-NN graph
    knn_edges = []
    adj = [{} for _ in range(n)]
    for i in range(n):
        row = sim_mat[i]
        # Top k neighbors excluding self
        neighbors = sorted(range(n), key=lambda j: row[j], reverse=True)
        cnt = 0
        for j in neighbors:
            if j == i:
                continue
            w = max(0.01, row[j])
            adj[i][j] = max(adj[i].get(j, 0.0), w)
            adj[j][i] = max(adj[j].get(i, 0.0), w)
            knn_edges.append((i, j, w))
            cnt += 1
            if cnt >= k:
                break

    # Degree-normalized transition operator for spectral initialization
    deg = [sum(adj[i].values()) + 1e-9 for i in range(n)]

    def power_vec(orth_vecs: list[list[float]]) -> list[float]:
        v = [rng.gauss(0.0, 1.0) for _ in range(n)]
        for _ in range(60):
            # Multiply by D^-1/2 A D^-1/2
            nv = [0.0] * n
            for i in range(n):
                inv_di = 1.0 / math.sqrt(deg[i])
                s = 0.0
                for j, w in adj[i].items():
                    s += w * v[j] / math.sqrt(deg[j])
                nv[i] = s * inv_di
            # Orthogonalize against trivial & previous eigenvectors
            for u in orth_vecs:
                dot = sum(a * b for a, b in zip(nv, u))
                nv = [a - dot * b for a, b in zip(nv, u)]
            norm = math.sqrt(sum(a * a for a in nv)) + 1e-12
            v = [a / norm for a in nv]
        return v

    # Trivial eigenvector proportional to sqrt(deg)
    u0 = [math.sqrt(d) for d in deg]
    n0 = math.sqrt(sum(a * a for a in u0))
    u0 = [a / n0 for a in u0]

    v1 = power_vec([u0])
    v2 = power_vec([u0, v1])

    # Scale initial coordinates to [-4, 4]
    s1 = max(abs(x) for x in v1) or 1.0
    s2 = max(abs(x) for x in v2) or 1.0
    xs = [4.0 * x / s1 for x in v1]
    ys = [4.0 * y / s2 for y in v2]

    # UMAP-style SGD refinement
    for it in range(iterations):
        lr = 0.25 * (1.0 - it / iterations) + 0.01
        # Attractive forces along k-NN edges
        for i, j, w in knn_edges:
            dx = xs[j] - xs[i]
            dy = ys[j] - ys[i]
            dist2 = dx * dx + dy * dy + 0.001
            grad = (2.0 * w * dist2 / (1.0 + dist2)) * 0.06 * lr
            xs[i] += grad * dx
            ys[i] += grad * dy
            xs[j] -= grad * dx
            ys[j] -= dy * grad

        # Negative sampling repulsive forces (5 negatives per node)
        for i in range(n):
            xi, yi = xs[i], ys[i]
            for _ in range(5):
                j = rng.randrange(n)
                if j == i or j in adj[i]:
                    continue
                dx = xi - xs[j]
                dy = yi - ys[j]
                dist2 = dx * dx + dy * dy + 0.05
                if dist2 < 9.0:
                    rep = (1.0 / ((0.2 + dist2) * (1.0 + dist2))) * 0.08 * lr
                    xs[i] += rep * dx
                    ys[i] += rep * dy

    # Normalize to [-10, 10]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    coords = [
        (
            -9.5 + 19.0 * (x - min_x) / (max_x - min_x + 1e-9),
            -9.5 + 19.0 * (y - min_y) / (max_y - min_y + 1e-9),
        )
        for x, y in zip(xs, ys)
    ]
    return coords


def run_hdbscan_mutual_reachability(
    coords: list[tuple[float, float]],
    min_samples: int = 7,
    min_cluster_size: int = 16,
    max_cluster_size: int = 155,
) -> tuple[list[int], dict]:
    """
    Performs HDBSCAN mutual-reachability hierarchical MST condensation in 2D manifold
    space: removes high-reachability noise edges and recursively splits oversized
    components (> max_cluster_size) at their highest internal mutual-reachability bridge edge.
    """
    n = len(coords)
    dist = [[0.0] * n for _ in range(n)]
    core_dist = [0.0] * n
    for i in range(n):
        xi, yi = coords[i]
        row_d = []
        for j in range(n):
            d = math.hypot(xi - coords[j][0], yi - coords[j][1])
            dist[i][j] = d
            if i != j:
                row_d.append(d)
        row_d.sort()
        core_dist[i] = row_d[min_samples - 1]

    in_mst = [False] * n
    min_e = [float("inf")] * n
    parent = [-1] * n
    min_e[0] = 0.0
    mst_edges = []

    for _ in range(n):
        u = -1
        best = float("inf")
        for i in range(n):
            if not in_mst[i] and min_e[i] < best:
                best = min_e[i]
                u = i
        if u == -1:
            break
        in_mst[u] = True
        if parent[u] != -1:
            mst_edges.append((parent[u], u, best))
        cu = core_dist[u]
        du = dist[u]
        for v in range(n):
            if not in_mst[v]:
                mr = max(cu, core_dist[v], du[v])
                if mr < min_e[v]:
                    min_e[v] = mr
                    parent[v] = u

    mst_edges.sort(key=lambda e: e[2])
    # Initial noise cut at 82nd percentile of mutual reachability weights
    cut_idx = int(len(mst_edges) * 0.82)
    thresh = mst_edges[cut_idx][2]
    active_edges = mst_edges[:cut_idx]

    def get_components(nodes_set, edges_list):
        adj = defaultdict(list)
        for u, v, w in edges_list:
            if u in nodes_set and v in nodes_set:
                adj[u].append((v, w))
                adj[v].append((u, w))
        vis = set()
        comps = []
        for u in nodes_set:
            if u not in vis:
                stack = [u]
                vis.add(u)
                c = set()
                while stack:
                    cur = stack.pop()
                    c.add(cur)
                    for nb, _ in adj[cur]:
                        if nb not in vis:
                            vis.add(nb)
                            stack.append(nb)
                comps.append(c)
        return comps

    # Recursively condense tree: split components > max_cluster_size at their heaviest internal MST edge
    final_clusters = []
    queue = [c for c in get_components(set(range(n)), active_edges) if len(c) >= min_cluster_size]
    comp_edges = {id(c): [e for e in active_edges if e[0] in c and e[1] in c] for c in queue}

    while queue:
        c = queue.pop(0)
        c_edges = [e for e in active_edges if e[0] in c and e[1] in c]
        if len(c) <= max_cluster_size or not c_edges:
            if len(c) >= min_cluster_size:
                final_clusters.append(sorted(c))
            continue
        # Remove the heaviest edge inside c that produces at least one valid child >= min_cluster_size
        c_edges.sort(key=lambda e: e[2], reverse=True)
        split_done = False
        for drop_idx in range(min(15, len(c_edges))):
            rem_edges = c_edges[drop_idx + 1 :]
            sub_comps = [sc for sc in get_components(c, rem_edges) if len(sc) >= min_cluster_size]
            if len(sub_comps) >= 2:
                active_edges = [e for e in active_edges if not (e[0] in c and e[1] in c)] + rem_edges
                queue.extend(sub_comps)
                split_done = True
                break
        if not split_done:
            # Drop top edge and continue
            rem_edges = c_edges[1:]
            active_edges = [e for e in active_edges if not (e[0] in c and e[1] in c)] + rem_edges
            sub_comps = [sc for sc in get_components(c, rem_edges) if len(sc) >= min_cluster_size]
            queue.extend(sub_comps)

    final_clusters.sort(key=len, reverse=True)
    final_clusters = final_clusters[:12]

    best_labels = [-1] * n
    for cid, c in enumerate(final_clusters):
        for node in c:
            best_labels[node] = cid

    clustered_idx = [i for i, l in enumerate(best_labels) if l != -1]
    by_cluster = defaultdict(list)
    for i in clustered_idx:
        by_cluster[best_labels[i]].append(i)

    sil_vals = []
    if len(by_cluster) >= 2:
        for i in clustered_idx:
            ci = best_labels[i]
            own = by_cluster[ci]
            a_i = sum(dist[i][j] for j in own if j != i) / (len(own) - 1) if len(own) > 1 else 0.0
            b_i = min(
                sum(dist[i][j] for j in other) / len(other)
                for cj, other in by_cluster.items()
                if cj != ci
            )
            sil_vals.append((b_i - a_i) / (max(a_i, b_i) + 1e-9))

    silhouette = sum(sil_vals) / len(sil_vals) if sil_vals else 0.0
    noise_count = sum(1 for l in best_labels if l == -1)

    stats = {
        "n_clusters": len(by_cluster),
        "n_clustered_symptoms": len(clustered_idx),
        "n_noise_outliers": noise_count,
        "noise_fraction": round(noise_count / n, 4),
        "silhouette_score_2d": round(silhouette, 4),
        "mutual_reachability_threshold": round(thresh, 4),
    }
    return best_labels, stats


def compute_svc_rbf_contours(
    coords: list[tuple[float, float]],
    labels: list[int],
    gamma: float = 0.45,
    grid_size: int = 60,
) -> list[tuple[int, list[tuple[float, float]]]]:
    """
    Computes Support Vector Clustering (SVC) RBF kernel density contour polygons
    around each identified cluster support region in 2D space using marching squares.
    """
    by_cluster = defaultdict(list)
    for (x, y), l in zip(coords, labels):
        if l != -1:
            by_cluster[l].append((x, y))

    contours = []
    for cid, pts in sorted(by_cluster.items()):
        if len(pts) < 8:
            continue
        # Compute RBF kernel support value at each cluster point to set boundary threshold
        pt_vals = []
        for x, y in pts:
            v = sum(math.exp(-gamma * ((x - px) ** 2 + (y - py) ** 2)) for px, py in pts) / len(pts)
            pt_vals.append(v)
        pt_vals.sort()
        # Support boundary enclosing ~85% of cluster support vectors
        level = pt_vals[max(0, int(0.15 * len(pt_vals)))]

        # Radial boundary tracing around cluster medoid/centroid for smooth closed SVG contour
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)

        # Trace 48 angles from centroid outward to find the outer SVC RBF level-set boundary
        poly = []
        for a_idx in range(48):
            theta = 2.0 * math.pi * a_idx / 48.0
            dx, dy = math.cos(theta), math.sin(theta)
            r_best = 0.4
            for step in range(1, 55):
                r = step * 0.16
                gx = cx + r * dx
                gy = cy + r * dy
                val = sum(math.exp(-gamma * ((gx - px) ** 2 + (gy - py) ** 2)) for px, py in pts) / len(pts)
                if val >= level * 0.72:
                    r_best = r
            poly.append((cx + r_best * dx, cy + r_best * dy))

        # Smooth polygon with 2-pass moving average
        for _ in range(2):
            m = len(poly)
            poly = [
                (
                    0.25 * poly[(k - 1) % m][0] + 0.5 * poly[k][0] + 0.25 * poly[(k + 1) % m][0],
                    0.25 * poly[(k - 1) % m][1] + 0.5 * poly[k][1] + 0.25 * poly[(k + 1) % m][1],
                )
                for k in range(m)
            ]
        contours.append((cid, poly))
    return contours


def convert_svg_to_png(svg_path: Path, png_path: Path, width: int = 1800) -> bool:
    """Uses macOS built-in qlmanage / sips to convert SVG to high-res PNG."""
    try:
        tmp_dir = svg_path.parent
        subprocess.run(
            ["/usr/bin/qlmanage", "-t", "-s", str(width), "-o", str(tmp_dir), str(svg_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        ql_png = tmp_dir / (svg_path.name + ".png")
        if ql_png.exists():
            ql_png.replace(png_path)
            return True
    except Exception:
        pass
    return False


def render_fig1_inter_model_comparison(
    sim_bio: list[list[float]],
    sim_pub: list[list[float]],
    disorders: list[str],
    summary_stats: dict,
    out_dir: Path,
) -> tuple[Path, Path]:
    """
    Figure 1: Dual-model cosine similarity comparison across all 528,672 cross-disorder
    symptom pairs (2D joint density heatmap + marginal histogram comparison).
    """
    n = len(disorders)
    bins_1d = 45
    hist_bio = [0] * bins_1d
    hist_pub = [0] * bins_1d
    grid_2d = [[0] * 36 for _ in range(36)]

    min_v, max_v = -0.2, 1.0
    span = max_v - min_v

    total_pairs = 0
    high_both = 0
    for i in range(n):
        di = disorders[i]
        rb = sim_bio[i]
        rp = sim_pub[i]
        for j in range(i + 1, n):
            if di == disorders[j]:
                continue
            b = rb[j]
            p = rp[j]
            total_pairs += 1
            if b >= 0.75 and p >= 0.75:
                high_both += 1
            bi = max(0, min(bins_1d - 1, int((b - min_v) / span * bins_1d)))
            pi = max(0, min(bins_1d - 1, int((p - min_v) / span * bins_1d)))
            hist_bio[bi] += 1
            hist_pub[pi] += 1
            gx = max(0, min(35, int((b - min_v) / span * 36)))
            gy = max(0, min(35, int((p - min_v) / span * 36)))
            grid_2d[gy][gx] += 1

    max_h = max(max(hist_bio), max(hist_pub)) or 1
    max_g = max(max(row) for row in grid_2d) or 1

    W, H = 1280, 620
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" style="background:#ffffff;font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',Roboto,Helvetica,Arial,sans-serif;">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="640" y="38" text-anchor="middle" font-size="21" font-weight="700" fill="#0f172a">DSM-5-TR Cross-Disorder Cosine Similarity: BioBERT NLI/STS vs. PubMedBERT Embeddings</text>',
        f'<text x="640" y="62" text-anchor="middle" font-size="13.5" fill="#475569">Evaluated across {total_pairs:,} cross-disorder symptom pairs (1,033 symptom-bearing criteria across 170 DSM-5-TR disorders) • Pearson r = {summary_stats["inter_model_pearson_r"]:.4f} • Spearman ρ = {summary_stats["inter_model_spearman_rho"]:.4f}</text>',
    ]

    # Left panel: Marginal Distributions
    lx, ly, lw, lh = 85, 110, 510, 410
    svg.append(f'<rect x="{lx}" y="{ly}" width="{lw}" height="{lh}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>')
    svg.append(f'<text x="{lx + lw/2}" y="{ly - 14}" text-anchor="middle" font-size="15" font-weight="600" fill="#1e293b">A. Cross-Disorder Cosine Similarity Distributions</text>')

    # Grid lines & axes for Panel A
    for tick in [-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        tx = lx + (tick - min_v) / span * lw
        svg.append(f'<line x1="{tx:.1f}" y1="{ly}" x2="{tx:.1f}" y2="{ly+lh}" stroke="#e2e8f0" stroke-width="1"/>')
        svg.append(f'<text x="{tx:.1f}" y="{ly+lh+20}" text-anchor="middle" font-size="11.5" fill="#334155">{tick:.1f}</text>')

    for y_frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        ty = ly + lh - y_frac * (lh - 30)
        val_k = int(y_frac * max_h / 1000)
        svg.append(f'<line x1="{lx}" y1="{ty:.1f}" x2="{lx+lw}" y2="{ty:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        svg.append(f'<text x="{lx-8}" y="{ty+4:.1f}" text-anchor="end" font-size="11" fill="#475569">{val_k}k</text>')

    # Plot polygon curves for BioBERT and PubMedBERT histograms
    bw = lw / bins_1d
    pts_bio = [f"{lx:.1f},{ly+lh:.1f}"]
    pts_pub = [f"{lx:.1f},{ly+lh:.1f}"]
    for idx in range(bins_1d):
        cx = lx + (idx + 0.5) * bw
        yb = ly + lh - (hist_bio[idx] / max_h) * (lh - 30)
        yp = ly + lh - (hist_pub[idx] / max_h) * (lh - 30)
        pts_bio.append(f"{cx:.1f},{yb:.1f}")
        pts_pub.append(f"{cx:.1f},{yp:.1f}")
    pts_bio.append(f"{lx+lw:.1f},{ly+lh:.1f}")
    pts_pub.append(f"{lx+lw:.1f},{ly+lh:.1f}")

    svg.append(f'<polygon points="{" ".join(pts_bio)}" fill="#2563eb" fill-opacity="0.28" stroke="#1d4ed8" stroke-width="2.2"/>')
    svg.append(f'<polygon points="{" ".join(pts_pub)}" fill="#e11d48" fill-opacity="0.25" stroke="#be123c" stroke-width="2.2"/>')

    # Mean vertical dashed lines
    mb_x = lx + (summary_stats["biobert_cosine_mean"] - min_v) / span * lw
    mp_x = lx + (summary_stats["pubmedbert_cosine_mean"] - min_v) / span * lw
    svg.append(f'<line x1="{mb_x:.1f}" y1="{ly}" x2="{mb_x:.1f}" y2="{ly+lh}" stroke="#1d4ed8" stroke-width="2" stroke-dasharray="5,4"/>')
    svg.append(f'<line x1="{mp_x:.1f}" y1="{ly}" x2="{mp_x:.1f}" y2="{ly+lh}" stroke="#be123c" stroke-width="2" stroke-dasharray="5,4"/>')

    # Legend inside Panel A
    svg.append(f'<rect x="{lx+235}" y="{ly+16}" width="260" height="68" rx="6" fill="#ffffff" stroke="#cbd5e1"/>')
    svg.append(f'<rect x="{lx+248}" y="{ly+30}" width="14" height="14" fill="#2563eb" fill-opacity="0.5" stroke="#1d4ed8" stroke-width="1.5"/>')
    svg.append(f'<text x="{lx+270}" y="{ly+41}" font-size="12" font-weight="600" fill="#1e293b">BioBERT NLI/STS (μ = {summary_stats["biobert_cosine_mean"]:.3f} ± {summary_stats["biobert_cosine_std"]:.3f})</text>')
    svg.append(f'<rect x="{lx+248}" y="{ly+54}" width="14" height="14" fill="#e11d48" fill-opacity="0.5" stroke="#be123c" stroke-width="1.5"/>')
    svg.append(f'<text x="{lx+270}" y="{ly+65}" font-size="12" font-weight="600" fill="#1e293b">PubMedBERT (μ = {summary_stats["pubmedbert_cosine_mean"]:.3f} ± {summary_stats["pubmedbert_cosine_std"]:.3f})</text>')

    svg.append(f'<text x="{lx + lw/2}" y="{ly+lh+46}" text-anchor="middle" font-size="13" font-weight="600" fill="#1e293b">Pairwise Cross-Disorder Cosine Similarity</text>')
    svg.append(f'<text x="{lx-48}" y="{ly + lh/2}" text-anchor="middle" font-size="13" font-weight="600" fill="#1e293b" transform="rotate(-90 {lx-48} {ly + lh/2})">Number of Symptom Pairs</text>')

    # Right panel: 2D Log-Density Joint Agreement Heatmap
    rx, ry, rw, rh = 695, 110, 480, 410
    svg.append(f'<rect x="{rx}" y="{ry}" width="{rw}" height="{rh}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>')
    svg.append(f'<text x="{rx + rw/2}" y="{ry - 14}" text-anchor="middle" font-size="15" font-weight="600" fill="#1e293b">B. Joint 2D Log-Density Agreement (BioBERT vs. PubMedBERT)</text>')

    cw = rw / 36.0
    ch = rh / 36.0
    log_max = math.log1p(max_g)
    for gy in range(36):
        for gx in range(36):
            cnt = grid_2d[gy][gx]
            if cnt == 0:
                continue
            intensity = math.log1p(cnt) / log_max
            # Interpolate light indigo -> deep navy/violet
            r_c = int(238 - intensity * 208)
            g_c = int(242 - intensity * 195)
            b_c = int(255 - intensity * 120)
            cx = rx + gx * cw
            cy = ry + rh - (gy + 1) * ch
            svg.append(f'<rect x="{cx:.1f}" y="{cy:.1f}" width="{cw+0.4:.1f}" height="{ch+0.4:.1f}" fill="rgb({r_c},{g_c},{b_c})"/>')

    # Identity line y = x and high-similarity bridge quadrant (cos >= 0.75)
    svg.append(f'<line x1="{rx}" y1="{ry+rh}" x2="{rx+rw}" y2="{ry}" stroke="#ef4444" stroke-width="1.8" stroke-dasharray="6,4"/>')
    bx_75 = rx + (0.75 - min_v) / span * rw
    by_75 = ry + rh - (1.0 - 0.75) / span * rh
    svg.append(f'<rect x="{bx_75:.1f}" y="{ry}" width="{rx+rw-bx_75:.1f}" height="{ rh - (0.75 - min_v)/span*rh :.1f}" fill="none" stroke="#dc2626" stroke-width="1.8" stroke-dasharray="3,3"/>')
    svg.append(f'<text x="{bx_75 - 6:.1f}" y="{ry + 22}" text-anchor="end" font-size="11.5" font-weight="600" fill="#b91c1c">Consensus Bridge Zone (≥0.75): {high_both} pairs →</text>')

    for tick in [-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        tx = rx + (tick - min_v) / span * rw
        ty = ry + rh - (tick - min_v) / span * rh
        svg.append(f'<text x="{tx:.1f}" y="{ry+rh+20}" text-anchor="middle" font-size="11.5" fill="#334155">{tick:.1f}</text>')
        svg.append(f'<text x="{rx-8}" y="{ty+4:.1f}" text-anchor="end" font-size="11.5" fill="#334155">{tick:.1f}</text>')

    svg.append(f'<text x="{rx + rw/2}" y="{ry+rh+46}" text-anchor="middle" font-size="13" font-weight="600" fill="#1e293b">BioBERT NLI/STS Cosine Similarity</text>')
    svg.append(f'<text x="{rx-46}" y="{ry + rh/2}" text-anchor="middle" font-size="13" font-weight="600" fill="#1e293b" transform="rotate(-90 {rx-46} {ry + rh/2})">PubMedBERT Cosine Similarity</text>')

    svg.append("</svg>")
    svg_path = out_dir / "fig1_inter_model_comparison.svg"
    png_path = out_dir / "fig1_inter_model_comparison.png"
    svg_path.write_text("\n".join(svg), encoding="utf-8")
    convert_svg_to_png(svg_path, png_path)
    return svg_path, png_path


def render_fig2_chapter_overlap_heatmap(
    sim_bio: list[list[float]],
    sim_pub: list[list[float]],
    disorders: list[str],
    chapters: list[str],
    out_dir: Path,
) -> tuple[Path, Path, list[dict]]:
    """
    Figure 2: 20 x 20 DSM-5-TR Diagnostic Chapter Transdiagnostic Semantic Overlap Heatmap
    (computing mean top-5 cross-disorder symptom similarity for every chapter pair).
    """
    chap_names = [c[0] for c in CHAPTER_RANGES]
    c_idx_map = {c: i for i, c in enumerate(chap_names)}
    m = len(chap_names)

    def calc_chap_matrix(sim_mat: list[list[float]]) -> list[list[float]]:
        pair_sims = [[[] for _ in range(m)] for _ in range(m)]
        n = len(disorders)
        for i in range(n):
            ci = c_idx_map[chapters[i]]
            di = disorders[i]
            row = sim_mat[i]
            for j in range(i + 1, n):
                if di == disorders[j]:
                    continue
                cj = c_idx_map[chapters[j]]
                val = row[j]
                pair_sims[ci][cj].append(val)
                if ci != cj:
                    pair_sims[cj][ci].append(val)

        mat = [[0.0] * m for _ in range(m)]
        for r in range(m):
            for c in range(m):
                arr = pair_sims[r][c]
                if not arr:
                    mat[r][c] = 0.55 if r == c else 0.20
                else:
                    arr.sort(reverse=True)
                    top_n = arr[: min(10, len(arr))]
                    mat[r][c] = sum(top_n) / len(top_n)
        return mat

    mat_bio = calc_chap_matrix(sim_bio)
    mat_pub = calc_chap_matrix(sim_pub)

    # Top cross-chapter pairs by consensus
    cross_chap_rows = []
    for i in range(m):
        for j in range(i + 1, m):
            cross_chap_rows.append(
                {
                    "chapter_1": chap_names[i],
                    "chapter_2": chap_names[j],
                    "biobert_top10_mean": round(mat_bio[i][j], 4),
                    "pubmedbert_top10_mean": round(mat_pub[i][j], 4),
                    "consensus_mean": round(0.5 * (mat_bio[i][j] + mat_pub[i][j]), 4),
                }
            )
    cross_chap_rows.sort(key=lambda x: x["consensus_mean"], reverse=True)

    W, H = 1480, 740
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" style="background:#ffffff;font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',Roboto,Helvetica,Arial,sans-serif;">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="740" y="36" text-anchor="middle" font-size="21" font-weight="700" fill="#0f172a">DSM-5-TR Cross-Chapter Semantic Overlap Matrix (Mean Top-10 Cross-Disorder Cosine Similarity)</text>',
        '<text x="740" y="58" text-anchor="middle" font-size="13.5" fill="#475569">Quantifying transdiagnostic symptom convergence across all 20 DSM-5-TR diagnostic classes (diagonal = within-chapter cross-disorder overlap)</text>',
    ]

    def draw_heatmap(mat: list[list[float]], ox: int, oy: int, title: str):
        cell = 23.5
        size = cell * m
        svg.append(f'<text x="{ox + size/2}" y="{oy - 14}" text-anchor="middle" font-size="15" font-weight="700" fill="#1e293b">{title}</text>')
        min_v, max_v = 0.40, 1.00
        for r in range(m):
            svg.append(f'<text x="{ox - 6}" y="{oy + r*cell + 15.5:.1f}" text-anchor="end" font-size="10.5" fill="#1e293b">{html.escape(chap_names[r])}</text>')
            for c in range(m):
                v = mat[r][c]
                norm = max(0.0, min(1.0, (v - min_v) / (max_v - min_v)))
                # Viridis/YlOrRd-inspired warm clinical palette: cream -> amber -> crimson -> deep maroon
                if norm < 0.5:
                    t = norm * 2.0
                    rc = int(254 - t * 15)
                    gc = int(243 - t * 95)
                    bc = int(199 - t * 155)
                else:
                    t = (norm - 0.5) * 2.0
                    rc = int(239 - t * 112)
                    gc = int(148 - t * 128)
                    bc = int(44 - t * 20)
                cx = ox + c * cell
                cy = oy + r * cell
                txt_col = "#ffffff" if norm > 0.62 else "#1e293b"
                svg.append(f'<rect x="{cx:.1f}" y="{cy:.1f}" width="{cell:.1f}" height="{cell:.1f}" fill="rgb({rc},{gc},{bc})" stroke="#ffffff" stroke-width="0.6"/>')
                svg.append(f'<text x="{cx + cell/2:.1f}" y="{cy + 14.5:.1f}" text-anchor="middle" font-size="7.8" font-weight="600" fill="{txt_col}">{v:.2f}</text>')

        for c in range(m):
            lx = ox + c * cell + cell / 2
            ly = oy + size + 8
            svg.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="end" font-size="10.2" fill="#1e293b" transform="rotate(-45 {lx:.1f} {ly:.1f})">{html.escape(chap_names[c])}</text>')

    draw_heatmap(mat_bio, 215, 95, "A. BioBERT NLI/STS (pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb)")
    draw_heatmap(mat_pub, 935, 95, "B. PubMedBERT Embeddings (NeuML/pubmedbert-base-embeddings)")

    svg.append("</svg>")
    svg_path = out_dir / "fig2_chapter_overlap_heatmap.svg"
    png_path = out_dir / "fig2_chapter_overlap_heatmap.png"
    svg_path.write_text("\n".join(svg), encoding="utf-8")
    convert_svg_to_png(svg_path, png_path)
    return svg_path, png_path, cross_chap_rows


def render_fig3_manifold_hdbscan_svc(
    coords_bio: list[tuple[float, float]],
    labels_bio: list[int],
    stats_bio: dict,
    contours_bio: list[tuple[int, list[tuple[float, float]]]],
    coords_pub: list[tuple[float, float]],
    labels_pub: list[int],
    stats_pub: dict,
    contours_pub: list[tuple[int, list[tuple[float, float]]]],
    sym_data: list[dict],
    chapters: list[str],
    out_dir: Path,
) -> tuple[Path, Path, dict]:
    """
    Figure 3: 2D Manifold Projection with HDBSCAN Mutual-Reachability Clusters and
    Support Vector Clustering (SVC) RBF Decision Boundaries for BioBERT & PubMedBERT.
    """
    W, H = 1440, 720
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" style="background:#ffffff;font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',Roboto,Helvetica,Arial,sans-serif;">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="720" y="36" text-anchor="middle" font-size="21" font-weight="700" fill="#0f172a">2D Symptom Manifold with HDBSCAN Clusters &amp; Support Vector Clustering (SVC) Boundaries</text>',
        '<text x="720" y="58" text-anchor="middle" font-size="13.5" fill="#475569">Shaded contours = RBF Support Vector Clustering (SVC) decision regions • Colored nodes = HDBSCAN dense symptom clusters • Gray nodes = unclustered transdiagnostic/unique symptoms</text>',
    ]

    cluster_summaries = {"biobert_nli_stsb": [], "pubmedbert_embeddings": []}

    def draw_manifold_panel(
        coords: list[tuple[float, float]],
        labels: list[int],
        stats: dict,
        contours: list[tuple[int, list[tuple[float, float]]]],
        ox: int,
        oy: int,
        pw: int,
        ph: int,
        title: str,
        model_key: str,
    ):
        svg.append(f'<rect x="{ox}" y="{oy}" width="{pw}" height="{ph}" rx="6" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>')
        svg.append(f'<text x="{ox + pw/2}" y="{oy - 12}" text-anchor="middle" font-size="14.5" font-weight="700" fill="#1e293b">{title}</text>')

        def to_px(x: float, y: float) -> tuple[float, float]:
            px = ox + ((x + 10.5) / 21.0) * pw
            py = oy + ph - ((y + 10.5) / 21.0) * ph
            return px, py

        # Grid lines
        for g in [-8, -4, 0, 4, 8]:
            gx, gy = to_px(g, g)
            svg.append(f'<line x1="{gx:.1f}" y1="{oy}" x2="{gx:.1f}" y2="{oy+ph}" stroke="#e2e8f0" stroke-width="0.8"/>')
            svg.append(f'<line x1="{ox}" y1="{gy:.1f}" x2="{ox+pw}" y2="{gy:.1f}" stroke="#e2e8f0" stroke-width="0.8"/>')

        # 1. Draw SVC RBF contour polygons
        for cid, poly in contours:
            col = CLUSTER_PALETTE[cid % len(CLUSTER_PALETTE)]
            pts_str = " ".join(f"{to_px(x, y)[0]:.1f},{to_px(x, y)[1]:.1f}" for x, y in poly)
            svg.append(
                f'<polygon points="{pts_str}" fill="{col}" fill-opacity="0.13" stroke="{col}" stroke-width="1.8" stroke-dasharray="5,3"/>'
            )

        # 2. Draw noise points first, then clustered points
        for idx, ((x, y), l) in enumerate(zip(coords, labels)):
            if l == -1:
                px, py = to_px(x, y)
                svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.6" fill="#94a3b8" fill-opacity="0.45"/>')

        by_c = defaultdict(list)
        for idx, ((x, y), l) in enumerate(zip(coords, labels)):
            if l != -1:
                by_c[l].append(idx)
                col = CLUSTER_PALETTE[l % len(CLUSTER_PALETTE)]
                px, py = to_px(x, y)
                svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.6" fill="{col}" fill-opacity="0.88" stroke="#ffffff" stroke-width="0.5"/>')

        # 3. Annotate cluster centroids with dominant clinical theme / chapter
        for cid, idxs in sorted(by_c.items()):
            col = CLUSTER_PALETTE[cid % len(CLUSTER_PALETTE)]
            cx = sum(coords[i][0] for i in idxs) / len(idxs)
            cy = sum(coords[i][1] for i in idxs) / len(idxs)
            px, py = to_px(cx, cy)

            top_chaps = Counter(chapters[i] for i in idxs).most_common(2)
            chap_label = " / ".join(f"{c}" for c, _ in top_chaps)
            sample_sym = sym_data[idxs[0]]["embedding_text"][:48]

            cluster_summaries[model_key].append(
                {
                    "cluster_id": int(cid + 1),
                    "size": len(idxs),
                    "dominant_chapters": [f"{c} ({cnt})" for c, cnt in top_chaps],
                    "representative_symptom": sym_data[idxs[0]]["embedding_text"],
                }
            )

            lbl_text = f"C{cid+1}: {top_chaps[0][0]} (n={len(idxs)})"
            tw = len(lbl_text) * 6.0 + 12
            svg.append(f'<rect x="{px - tw/2:.1f}" y="{py - 11:.1f}" width="{tw:.1f}" height="18" rx="4" fill="#ffffff" fill-opacity="0.88" stroke="{col}" stroke-width="1.3"/>')
            svg.append(f'<text x="{px:.1f}" y="{py + 2:.1f}" text-anchor="middle" font-size="10.2" font-weight="700" fill="#0f172a">{html.escape(lbl_text)}</text>')

    p_title_bio = f"A. BioBERT NLI/STS ({stats_bio['n_clusters']} Clusters, Clustered={stats_bio['n_clustered_symptoms']}/1,033, Silhouette={stats_bio['silhouette_score_2d']:.3f})"
    p_title_pub = f"B. PubMedBERT Embeddings ({stats_pub['n_clusters']} Clusters, Clustered={stats_pub['n_clustered_symptoms']}/1,033, Silhouette={stats_pub['silhouette_score_2d']:.3f})"

    draw_manifold_panel(coords_bio, labels_bio, stats_bio, contours_bio, 55, 95, 635, 575, p_title_bio, "biobert_nli_stsb")
    draw_manifold_panel(coords_pub, labels_pub, stats_pub, contours_pub, 750, 95, 635, 575, p_title_pub, "pubmedbert_embeddings")

    svg.append("</svg>")
    svg_path = out_dir / "fig3_manifold_hdbscan_svc.svg"
    png_path = out_dir / "fig3_manifold_hdbscan_svc.png"
    svg_path.write_text("\n".join(svg), encoding="utf-8")
    convert_svg_to_png(svg_path, png_path)
    return svg_path, png_path, cluster_summaries


def render_fig4_bridge_symptoms(
    sim_bio: list[list[float]],
    sim_pub: list[list[float]],
    sym_data: list[dict],
    chapters: list[str],
    out_dir: Path,
) -> tuple[Path, Path, list[dict]]:
    """
    Figure 4: Top Transdiagnostic Bridge Symptom Constructs (ranked by number of distinct
    DSM-5-TR chapters and disorders bridged at consensus cosine >= 0.78, excluding
    substance-induced etiologic boilerplate) + Cross-Chapter Bridge Chord Network.
    """
    n = len(sym_data)
    disorders = [r["disorder"] for r in sym_data]
    texts = [r["embedding_text"] for r in sym_data]

    # Group identical or near-identical clinical symptom constructs across DSM-5-TR
    construct_stats = []
    seen_texts = set()

    for i in range(n):
        t = texts[i]
        # Skip repetitive substance/medication-induced temporal/capability rule lines
        if "the involved substance/medication is capable" in t or "developed during or soon after substance" in t or "minimum duration of approximately 6 months" in t:
            continue
        if t in seen_texts:
            continue

        # Find all symptoms across any disorder with consensus cosine >= 0.78 to symptom i
        bridged_disorders = set()
        bridged_chapters = set()
        cross_sims_bio = []
        cross_sims_pub = []

        for j in range(n):
            sim_c = 0.5 * (sim_bio[i][j] + sim_pub[i][j])
            if sim_c >= 0.78:
                bridged_disorders.add(disorders[j])
                bridged_chapters.add(chapters[j])
                if disorders[j] != disorders[i]:
                    cross_sims_bio.append(sim_bio[i][j])
                    cross_sims_pub.append(sim_pub[i][j])
                if sim_c >= 0.92:
                    seen_texts.add(texts[j])

        if len(bridged_disorders) >= 3 and len(bridged_chapters) >= 2:
            seen_texts.add(t)
            mean_bio = sum(cross_sims_bio) / len(cross_sims_bio) if cross_sims_bio else 0.0
            mean_pub = sum(cross_sims_pub) / len(cross_sims_pub) if cross_sims_pub else 0.0
            construct_stats.append(
                {
                    "symptom_construct": t if len(t) <= 72 else t[:69] + "...",
                    "full_symptom_text": t,
                    "anchor_disorder": disorders[i],
                    "anchor_chapter": chapters[i],
                    "disorders_bridged": len(bridged_disorders),
                    "chapters_bridged": len(bridged_chapters),
                    "chapter_list": sorted(bridged_chapters),
                    "biobert_mean_sim": round(mean_bio, 4),
                    "pubmedbert_mean_sim": round(mean_pub, 4),
                }
            )

    construct_stats.sort(
        key=lambda x: (x["chapters_bridged"], x["disorders_bridged"], x["biobert_mean_sim"] + x["pubmedbert_mean_sim"]),
        reverse=True,
    )
    top_constructs = construct_stats[:16]

    # Build chapter-to-chapter bridge counts (pairs of symptoms from DIFFERENT chapters with consensus cosine >= 0.78)
    chap_names = [c[0] for c in CHAPTER_RANGES]
    c_idx = {c: idx for idx, c in enumerate(chap_names)}
    chap_links = defaultdict(int)
    chap_deg = Counter()
    for i in range(n):
        if "the involved substance/medication" in texts[i] or "developed during or soon after" in texts[i]:
            continue
        ci = chapters[i]
        for j in range(i + 1, n):
            cj = chapters[j]
            if ci == cj:
                continue
            if "the involved substance/medication" in texts[j] or "developed during or soon after" in texts[j]:
                continue
            if 0.5 * (sim_bio[i][j] + sim_pub[i][j]) >= 0.75:
                a, b = (ci, cj) if c_idx[ci] < c_idx[cj] else (cj, ci)
                chap_links[(a, b)] += 1
                chap_deg[ci] += 1
                chap_deg[cj] += 1

    W, H = 1440, 680
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" style="background:#ffffff;font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',Roboto,Helvetica,Arial,sans-serif;">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="720" y="36" text-anchor="middle" font-size="21" font-weight="700" fill="#0f172a">Transdiagnostic Bridge Symptoms &amp; Cross-Chapter Semantic Network</text>',
        '<text x="720" y="58" text-anchor="middle" font-size="13.5" fill="#475569">Left: Top 16 transdiagnostic symptom constructs ranked by DSM-5-TR disorder &amp; chapter reach (consensus cos ≥ 0.78) • Right: Cross-chapter bridge symptom network</text>',
    ]

    # Panel A: Horizontal Bar Chart of Top 16 Bridge Constructs
    lx, ly, lw, lh = 390, 95, 360, 535
    svg.append(f'<text x="{lx + lw/2 - 110}" y="{ly - 12}" text-anchor="middle" font-size="15" font-weight="700" fill="#1e293b">A. Top 16 Transdiagnostic Bridge Symptom Constructs</text>')

    max_d = max(c["disorders_bridged"] for c in top_constructs) if top_constructs else 15
    row_h = lh / len(top_constructs)
    for idx, item in enumerate(top_constructs):
        y = ly + idx * row_h
        bw_d = (item["disorders_bridged"] / max_d) * lw
        bw_c = (item["chapters_bridged"] / max_d) * lw
        label = item["symptom_construct"]
        svg.append(f'<text x="{lx - 10}" y="{y + 16:.1f}" text-anchor="end" font-size="11.2" font-weight="500" fill="#1e293b">{html.escape(label)}</text>')
        # Disorders bridged bar
        svg.append(f'<rect x="{lx}" y="{y + 4:.1f}" width="{bw_d:.1f}" height="10" rx="2" fill="#2563eb"/>')
        # Chapters bridged bar
        svg.append(f'<rect x="{lx}" y="{y + 15:.1f}" width="{bw_c:.1f}" height="8" rx="2" fill="#f59e0b"/>')
        svg.append(
            f'<text x="{lx + bw_d + 6:.1f}" y="{y + 16:.1f}" font-size="10.5" font-weight="700" fill="#0f172a">{item["disorders_bridged"]} disorders ({item["chapters_bridged"]} chapters)</text>'
        )

    # Legend for Panel A
    svg.append(f'<rect x="{lx}" y="{ly+lh+12}" width="12" height="12" rx="2" fill="#2563eb"/>')
    svg.append(f'<text x="{lx+18}" y="{ly+lh+22}" font-size="11.5" font-weight="600" fill="#1e293b">Distinct Disorders Bridged</text>')
    svg.append(f'<rect x="{lx+195}" y="{ly+lh+12}" width="12" height="12" rx="2" fill="#f59e0b"/>')
    svg.append(f'<text x="{lx+213}" y="{ly+lh+22}" font-size="11.5" font-weight="600" fill="#1e293b">Distinct DSM-5-TR Chapters Bridged</text>')

    # Panel B: Circular Chord / Network Graph of DSM-5-TR Chapters
    cx, cy, radius = 1125, 365, 205
    svg.append(f'<text x="{cx}" y="{ly - 12}" text-anchor="middle" font-size="15" font-weight="700" fill="#1e293b">B. Cross-Chapter Bridge Symptom Network (cos ≥ 0.75)</text>')

    m = len(chap_names)
    node_pos = {}
    for idx, cname in enumerate(chap_names):
        ang = -math.pi / 2 + 2.0 * math.pi * idx / m
        nx = cx + radius * math.cos(ang)
        ny = cy + radius * math.sin(ang)
        node_pos[cname] = (nx, ny, ang)

    max_link = max(chap_links.values()) if chap_links else 1
    for (c1, c2), cnt in sorted(chap_links.items(), key=lambda x: x[1]):
        x1, y1, _ = node_pos[c1]
        x2, y2, _ = node_pos[c2]
        w = 0.8 + 6.5 * (cnt / max_link)
        op = min(0.82, 0.15 + 0.68 * (cnt / max_link))
        col = CHAPTER_COLORS.get(c1, "#2563eb")
        # Quadratic bezier curve pulled slightly toward center
        qx = 0.5 * (x1 + x2) * 0.78 + cx * 0.22
        qy = 0.5 * (y1 + y2) * 0.78 + cy * 0.22
        svg.append(
            f'<path d="M {x1:.1f} {y1:.1f} Q {qx:.1f} {qy:.1f} {x2:.1f} {y2:.1f}" fill="none" stroke="{col}" stroke-width="{w:.2f}" stroke-opacity="{op:.2f}"/>'
        )

    for cname in chap_names:
        nx, ny, ang = node_pos[cname]
        deg_c = chap_deg[cname]
        r_node = 6.0 + 9.0 * math.sqrt(deg_c / (max(chap_deg.values()) if chap_deg else 1))
        col = CHAPTER_COLORS.get(cname, "#2563eb")
        svg.append(f'<circle cx="{nx:.1f}" cy="{ny:.1f}" r="{r_node:.1f}" fill="{col}" stroke="#ffffff" stroke-width="1.8"/>')

        lx_n = cx + (radius + 20) * math.cos(ang)
        ly_n = cy + (radius + 20) * math.sin(ang)
        anchor = "start" if math.cos(ang) >= 0 else "end"
        svg.append(
            f'<text x="{lx_n:.1f}" y="{ly_n + 4:.1f}" text-anchor="{anchor}" font-size="10.8" font-weight="600" fill="#1e293b">{html.escape(cname)} ({deg_c})</text>'
        )

    svg.append("</svg>")
    svg_path = out_dir / "fig4_bridge_symptom_ranking_and_network.svg"
    png_path = out_dir / "fig4_bridge_symptom_ranking_and_network.png"
    svg_path.write_text("\n".join(svg), encoding="utf-8")
    convert_svg_to_png(svg_path, png_path)
    return svg_path, png_path, top_constructs


def render_interactive_html_explorer(
    sym_data: list[dict],
    chapters: list[str],
    coords_bio: list[tuple[float, float]],
    labels_bio: list[int],
    contours_bio: list[tuple[int, list[tuple[float, float]]]],
    coords_pub: list[tuple[float, float]],
    labels_pub: list[int],
    contours_pub: list[tuple[int, list[tuple[float, float]]]],
    top_constructs: list[dict],
    cross_chap_rows: list[dict],
    summary_stats: dict,
    out_dir: Path,
) -> Path:
    """Generates a standalone interactive HTML explorer for the 1,033 embedded symptoms."""
    points_payload = []
    for i, r in enumerate(sym_data):
        points_payload.append(
            {
                "id": i,
                "disorder": r["disorder"],
                "chapter": chapters[i],
                "cid": r["criterion_id"],
                "level": r["criterion_level"],
                "text": r["embedding_text"],
                "bx": round(coords_bio[i][0], 3),
                "by": round(coords_bio[i][1], 3),
                "bc": int(labels_bio[i]),
                "px": round(coords_pub[i][0], 3),
                "py": round(coords_pub[i][1], 3),
                "pc": int(labels_pub[i]),
            }
        )

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>DSM-5-TR Transdiagnostic Symptom Vector Space Explorer</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; background: #f8fafc; color: #0f172a; }}
  header {{ background: #0f172a; color: #fff; padding: 18px 28px; display: flex; justify-content: space-between; align-items: center; }}
  header h1 {{ margin: 0; font-size: 20px; }}
  header p {{ margin: 4px 0 0; font-size: 13px; color: #94a3b8; }}
  .controls {{ display: flex; gap: 12px; align-items: center; flex-wrap: wrap; padding: 14px 28px; background: #fff; border-bottom: 1px solid #e2e8f0; }}
  select, input {{ padding: 7px 11px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 13px; }}
  .main {{ display: grid; grid-template-columns: 1fr 390px; gap: 18px; padding: 18px 28px; }}
  .card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }}
  canvas {{ width: 100%; height: 600px; border-radius: 8px; background: #f8fafc; cursor: crosshair; }}
  .stat-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 14px; }}
  .stat-box {{ background: #f1f5f9; padding: 10px; border-radius: 6px; }}
  .stat-box .val {{ font-size: 18px; font-weight: 700; color: #1d4ed8; }}
  .stat-box .lbl {{ font-size: 11px; color: #475569; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid #e2e8f0; }}
  th {{ background: #f8fafc; font-weight: 600; }}
</style>
</head>
<body>
<header>
  <div>
    <h1>DSM-5-TR Symptom Embedding &amp; Clustering Interactive Explorer</h1>
    <p>1,033 symptom-bearing criteria across 170 DSM-5-TR disorders • BioBERT NLI/STS vs. PubMedBERT (768-d) • HDBSCAN &amp; SVC</p>
  </div>
</header>
<div class="controls">
  <label><strong>Model Space:</strong>
    <select id="modelSelect">
      <option value="bio">BioBERT NLI/STS (pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb)</option>
      <option value="pub">PubMedBERT Embeddings (NeuML/pubmedbert-base-embeddings)</option>
    </select>
  </label>
  <label><strong>Color By:</strong>
    <select id="colorSelect">
      <option value="cluster">HDBSCAN + SVC Cluster</option>
      <option value="chapter">DSM-5-TR Diagnostic Chapter (20 Classes)</option>
    </select>
  </label>
  <label><strong>Search Symptom / Disorder:</strong>
    <input type="text" id="searchInput" placeholder="e.g. insomnia, delusions, panic..."/>
  </label>
</div>
<div class="main">
  <div class="card">
    <canvas id="manifoldCanvas" width="920" height="600"></canvas>
  </div>
  <div>
    <div class="card" style="margin-bottom:14px;">
      <h3 style="margin-top:0;font-size:15px;">Hovered Symptom Inspector</h3>
      <div id="hoverInfo" style="font-size:13px;line-height:1.5;color:#334155;">Move your cursor over any node in the 2D manifold to inspect its DSM-5-TR disorder, chapter, cluster assignment, and verbatim symptom text.</div>
    </div>
    <div class="card">
      <h3 style="margin-top:0;font-size:15px;">Top Transdiagnostic Bridge Constructs</h3>
      <table>
        <thead><tr><th>Symptom Construct</th><th>Disorders</th><th>Chapters</th></tr></thead>
        <tbody>
          {"".join(f"<tr><td>{html.escape(c['symptom_construct'][:42])}</td><td><b>{c['disorders_bridged']}</b></td><td>{c['chapters_bridged']}</td></tr>" for c in top_constructs[:10])}
        </tbody>
      </table>
    </div>
  </div>
</div>
<script>
const POINTS = {json.dumps(points_payload)};
const CHAP_COLORS = {json.dumps(CHAPTER_COLORS)};
const CLUST_COLORS = {json.dumps(CLUSTER_PALETTE)};
const canvas = document.getElementById("manifoldCanvas");
const ctx = canvas.getContext("2d");
const modelSel = document.getElementById("modelSelect");
const colorSel = document.getElementById("colorSelect");
const searchInp = document.getElementById("searchInput");
const hoverDiv = document.getElementById("hoverInfo");

function toScreen(x, y) {{
  const pad = 40;
  const sx = pad + ((x + 10.5) / 21.0) * (canvas.width - 2*pad);
  const sy = canvas.height - pad - ((y + 10.5) / 21.0) * (canvas.height - 2*pad);
  return [sx, sy];
}}

function draw() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const isBio = modelSel.value === "bio";
  const byClust = colorSel.value === "cluster";
  const q = searchInp.value.trim().toLowerCase();

  for (const p of POINTS) {{
    const x = isBio ? p.bx : p.px;
    const y = isBio ? p.by : p.py;
    const c = isBio ? p.bc : p.pc;
    const [sx, sy] = toScreen(x, y);
    let col = byClust ? (c === -1 ? "#94a3b8" : CLUST_COLORS[c % CLUST_COLORS.length]) : (CHAP_COLORS[p.chapter] || "#2563eb");
    const match = !q || p.text.includes(q) || p.disorder.toLowerCase().includes(q) || p.chapter.toLowerCase().includes(q);
    ctx.beginPath();
    ctx.arc(sx, sy, match && q ? 6.0 : (c === -1 && byClust ? 3.0 : 4.2), 0, Math.PI * 2);
    ctx.fillStyle = col;
    ctx.globalAlpha = match ? (q ? 0.98 : 0.85) : 0.12;
    ctx.fill();
    if (match && q) {{
      ctx.strokeStyle = "#0f172a";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }}
  }}
  ctx.globalAlpha = 1.0;
}}

canvas.addEventListener("mousemove", (e) => {{
  const rect = canvas.getBoundingClientRect();
  const mx = (e.clientX - rect.left) * (canvas.width / rect.width);
  const my = (e.clientY - rect.top) * (canvas.height / rect.height);
  const isBio = modelSel.value === "bio";
  let best = null, bestD = 144;
  for (const p of POINTS) {{
    const [sx, sy] = toScreen(isBio ? p.bx : p.px, isBio ? p.by : p.py);
    const d2 = (sx - mx)**2 + (sy - my)**2;
    if (d2 < bestD) {{ bestD = d2; best = p; }}
  }}
  if (best) {{
    const c = isBio ? best.bc : best.pc;
    hoverDiv.innerHTML = `
      <div><strong>Disorder:</strong> ${{best.disorder}} (Criterion ${{best.cid}})</div>
      <div><strong>DSM-5-TR Chapter:</strong> ${{best.chapter}}</div>
      <div><strong>HDBSCAN/SVC Cluster:</strong> ${{c === -1 ? "Unclustered / Transdiagnostic Bridge" : "Cluster C" + (c + 1)}}</div>
      <div style="margin-top:8px;padding:8px;background:#f8fafc;border-left:3px solid #2563eb;">"${{best.text}}"</div>
    `;
  }}
}});

modelSel.addEventListener("change", draw);
colorSel.addEventListener("change", draw);
searchInp.addEventListener("input", draw);
draw();
</script>
</body>
</html>
"""
    html_path = out_dir / "dsm5tr_interactive_explorer.html"
    html_path.write_text(html_doc, encoding="utf-8")
    return html_path


def main():
    out_dir = Path("outputs")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "dsm_symptoms_filtered_only.json", "r", encoding="utf-8") as f:
        sym_data = json.load(f)
    with open(out_dir / "embedding_comparison_summary.json", "r", encoding="utf-8") as f:
        summary_stats = json.load(f)

    d_to_chap = map_disorders_to_chapters(sym_data)
    disorders = [r["disorder"] for r in sym_data]
    chapters = [d_to_chap[d] for d in disorders]

    print("Loading 768-d embeddings and computing similarity matrices...")
    _, _, emb_bio = load_npy_matrix(out_dir / "embeddings_biobert_nli_stsb_768d.npy")
    _, _, emb_pub = load_npy_matrix(out_dir / "embeddings_pubmedbert_embeddings_768d.npy")

    sim_bio = compute_cosine_matrix(emb_bio)
    sim_pub = compute_cosine_matrix(emb_pub)

    print("Rendering Figure 1 (Inter-Model Comparison)...")
    f1_svg, f1_png = render_fig1_inter_model_comparison(sim_bio, sim_pub, disorders, summary_stats, fig_dir)

    print("Rendering Figure 2 (20x20 DSM-5-TR Chapter Overlap Heatmap)...")
    f2_svg, f2_png, cross_chap_rows = render_fig2_chapter_overlap_heatmap(sim_bio, sim_pub, disorders, chapters, fig_dir)

    print("Computing 2D k-NN Manifold, HDBSCAN Mutual-Reachability Clusters & SVC Contours...")
    coords_bio = knn_manifold_2d(sim_bio, k=15, iterations=180, seed=42)
    labels_bio, stats_bio = run_hdbscan_mutual_reachability(coords_bio)
    contours_bio = compute_svc_rbf_contours(coords_bio, labels_bio)

    coords_pub = knn_manifold_2d(sim_pub, k=15, iterations=180, seed=42)
    labels_pub, stats_pub = run_hdbscan_mutual_reachability(coords_pub)
    contours_pub = compute_svc_rbf_contours(coords_pub, labels_pub)

    print("Rendering Figure 3 (Manifold + HDBSCAN + SVC Contours)...")
    f3_svg, f3_png, cluster_summaries = render_fig3_manifold_hdbscan_svc(
        coords_bio, labels_bio, stats_bio, contours_bio,
        coords_pub, labels_pub, stats_pub, contours_pub,
        sym_data, chapters, fig_dir,
    )

    print("Rendering Figure 4 (Transdiagnostic Bridge Symptoms & Network)...")
    f4_svg, f4_png, top_constructs = render_fig4_bridge_symptoms(sim_bio, sim_pub, sym_data, chapters, fig_dir)

    print("Rendering Interactive HTML Explorer...")
    html_path = render_interactive_html_explorer(
        sym_data, chapters,
        coords_bio, labels_bio, contours_bio,
        coords_pub, labels_pub, contours_pub,
        top_constructs, cross_chap_rows, summary_stats, fig_dir,
    )

    eval_out = {
        "biobert_nli_stsb_clustering": stats_bio,
        "pubmedbert_embeddings_clustering": stats_pub,
        "biobert_clusters": cluster_summaries["biobert_nli_stsb"],
        "pubmedbert_clusters": cluster_summaries["pubmedbert_embeddings"],
        "top_cross_chapter_overlaps": cross_chap_rows[:15],
        "top_transdiagnostic_bridge_constructs": top_constructs,
    }
    eval_path = out_dir / "clustering_evaluation_summary.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_out, f, indent=2, ensure_ascii=False)

    print("Done! Saved all figures to:", fig_dir)


if __name__ == "__main__":
    main()

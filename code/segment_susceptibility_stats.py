#!/usr/bin/env python3
"""
Computes per-segment landslide susceptibility statistics for the Nilgiri
Mountain Railway corridor.

Every susceptibility raster pixel inside the 1,000 m corridor buffer is assigned
to the nearest of the 11 inter-station segments, and the mean, median, and rank
of the predicted probability are reported per segment per model.

Probabilities are raw for BLR, RF, and XGBoost, and Platt-calibrated for SVM.
Probability magnitudes are not directly comparable across models.

Required input files:
    vector_data/buffer_boundary.shp
    vector_data/railway_line.shp
    vector_data/station_points.shp
    susceptibility_maps/susceptibility_{blr,svm,rf,xgb}_probability.tif
    statistics/manuscript_model_metrics.json
    statistics/xgb_model_metrics.json

Output file:
    statistics/segment_susceptibility_stats.csv

Requirements:
    pip install rasterio geopandas numpy pandas scipy shapely

Run:
    python segment_susceptibility_stats.py
"""

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask
from scipy.spatial import cKDTree
from shapely.geometry import Point
from shapely.ops import linemerge

ROOT = Path(__file__).resolve().parent.parent

VECTOR_DIR = ROOT / "vector_data"
MAPS_DIR = ROOT / "susceptibility_maps"
STATS_DIR = ROOT / "statistics"

BUFFER_SHP = VECTOR_DIR / "buffer_boundary.shp"
RAIL_SHP = VECTOR_DIR / "railway_line.shp"
STATION_SHP = VECTOR_DIR / "station_points.shp"

RASTERS = {
    "Binary logistic regression": MAPS_DIR / "susceptibility_blr_probability.tif",
    "Support vector machine": MAPS_DIR / "susceptibility_svm_probability.tif",
    "Random forest": MAPS_DIR / "susceptibility_rf_probability.tif",
    "XGBoost": MAPS_DIR / "susceptibility_xgb_probability.tif",
}

# Model keys in the metrics files, used to order models by test ROC-AUC.
METRICS_FILES = [
    STATS_DIR / "manuscript_model_metrics.json",
    STATS_DIR / "xgb_model_metrics.json",
]
MODEL_KEYS = {
    "BLR": "Binary logistic regression",
    "SVM": "Support vector machine",
    "RF": "Random forest",
    "XGB": "XGBoost",
}

# Station sequence, south (Mettupalayam) to north (Udagamandalam).
STATIONS = [
    "MTP",
    "QLR",
    "ADY",
    "HLG",
    "RME",
    "ONR",
    "WEL",
    "AVK",
    "KXT",
    "LOV",
    "FNHL",
    "UAM",
]
SEGMENTS = [f"{a}-{b}" for a, b in zip(STATIONS, STATIONS[1:], strict=False)]

OUT_CSV = STATS_DIR / "segment_susceptibility_stats.csv"

TARGET_CRS = "EPSG:32643"
LINE_STEP_M = 10.0


# Helpers


def check_inputs():
    """Verify every input file exists and every raster is in TARGET_CRS."""
    missing = [
        p
        for p in [BUFFER_SHP, RAIL_SHP, STATION_SHP, *RASTERS.values()]
        if not p.exists()
    ]
    if missing:
        raise SystemExit(
            "Missing input files:\n" + "\n".join(f"  {p}" for p in missing)
        )

    # Pixel centres are compared with the centreline without reprojection.
    for name, path in RASTERS.items():
        with rasterio.open(path) as src:
            if src.crs is None or src.crs.to_string() != TARGET_CRS:
                raise SystemExit(f"{name} raster is {src.crs}, expected {TARGET_CRS}.")


def model_order():
    """Order models by descending test ROC-AUC, read from the metrics files."""
    auc = {}
    for path in METRICS_FILES:
        if not path.exists():
            continue
        obj = json.loads(path.read_text())
        models = obj.get("models", obj)
        for key, name in MODEL_KEYS.items():
            if isinstance(models.get(key), dict) and "roc_auc" in models[key]:
                auc[name] = float(models[key]["roc_auc"])

    if set(auc) != set(RASTERS):
        raise SystemExit(
            "The bundled manuscript and XGBoost metrics must cover all four models."
        )

    order = sorted(auc, key=auc.get, reverse=True)
    print("Model order (descending test ROC-AUC)")
    for name in order:
        print(f"  {name:<28} {auc[name]:.3f}")
    return order


def load_vectors():
    """Load the buffer, centreline, and station points in TARGET_CRS."""
    layers = [gpd.read_file(p) for p in (BUFFER_SHP, RAIL_SHP, STATION_SHP)]
    for g in layers:
        if g.crs is None:
            raise ValueError(
                "Vector CRS is missing; define it before calculating distances."
            )
    return [g.to_crs(TARGET_CRS) for g in layers]


def station_code_field(stations):
    """Return the station-code column, allowing for DBF name truncation."""
    for col in stations.columns:
        key = col.lower().replace(" ", "").replace("_", "")
        if key.startswith("stationco") or (key.startswith("station") and "code" in key):
            return col
    for fallback in ("code", "CODE", "NAME", "Name", "STATION", "Station"):
        if fallback in stations.columns:
            return fallback
    raise SystemExit(f"No station-code column. Columns: {list(stations.columns)}")


def centreline(rail):
    """Merge the railway geometries into a single LineString."""
    geoms = list(rail.geometry)
    line = linemerge(geoms) if len(geoms) > 1 else geoms[0]

    if line.geom_type == "MultiLineString":
        raise ValueError(
            "Railway parts are disconnected; supply one continuous centreline."
        )
    return line


def station_chainage(line, stations, code_field):
    """Chainage of each station along the centreline, in metres from MTP."""
    chainage = {
        str(row[code_field]).strip(): float(
            line.project(Point(row.geometry.x, row.geometry.y))
        )
        for _, row in stations.iterrows()
    }

    missing = [s for s in STATIONS if s not in chainage]
    if missing:
        raise SystemExit(f"Station points missing: {missing}")

    print("\nStation chainage (m from MTP)")
    for station in STATIONS:
        print(f"  {station:<5} {chainage[station]:10,.1f}")

    # Segment labels are built from STATIONS, so chainage must increase in that
    # order. If it does not, pixels in the affected stretch would be assigned
    # labels that do not exist and would be dropped without an error.
    reversed_pairs = [
        (a, b)
        for a, b in zip(STATIONS, STATIONS[1:], strict=False)
        if chainage[b] <= chainage[a]
    ]
    if reversed_pairs:
        raise SystemExit(
            f"Chainage is not monotonic along the station sequence: {reversed_pairs}. "
            f"Check that the centreline is digitised continuously from Mettupalayam "
            f"to Udagamandalam and that the station points snap to it."
        )
    return chainage


def densify(line, step=LINE_STEP_M):
    """Sample the centreline every `step` metres."""
    n = int(np.ceil(line.length / step)) + 1
    chainages = np.linspace(0.0, line.length, n)
    xy = np.array([[p.x, p.y] for p in (line.interpolate(c) for c in chainages)])
    return xy, chainages


def segment_of(chainage_m, chainage):
    """Segment label containing a chainage, or None if outside the range."""
    for a, b in zip(STATIONS, STATIONS[1:], strict=False):
        if chainage[a] <= chainage_m <= chainage[b]:
            return f"{a}-{b}"
    return None


def clipped_pixels(path, buffer_geoms):
    """Coordinates and values of every valid raster pixel inside the buffer."""
    with rasterio.open(path) as src:
        nodata = src.nodata
        arr, transform = rio_mask(
            src,
            buffer_geoms,
            crop=True,
            filled=True,
            nodata=nodata if nodata is not None else np.nan,
        )
        data = arr[0].astype("float32")

    if nodata is not None and np.isfinite(nodata):
        data = np.where(data == nodata, np.nan, data)

    rows, cols = np.where(np.isfinite(data))
    xs = transform.c + (cols + 0.5) * transform.a + (rows + 0.5) * transform.b
    ys = transform.f + (cols + 0.5) * transform.d + (rows + 0.5) * transform.e
    return np.column_stack([xs, ys]), data[rows, cols]


def plot_segment_means(df, output):
    """Plot segment means on a common probability scale."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    names = ["Random forest", "Binary logistic regression", "XGBoost", "Support vector machine"]
    values = df.pivot(index="segment", columns="model", values="mean_prob").loc[SEGMENTS, names]
    cmap = LinearSegmentedColormap.from_list(
        "susceptibility", ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]
    )
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    mesh = ax.imshow(values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4), ["RF", "BLR", "XGB", "SVM"])
    ax.set_yticks(range(len(SEGMENTS)), SEGMENTS)
    ax.set_xlabel("Model")
    ax.set_ylabel("Segment (south to north)")
    ax.tick_params(labelsize=8)
    for (row, column), value in np.ndenumerate(values.to_numpy()):
        ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=7,
                color="white" if value < 0.25 or value > 0.8 else "black")
    fig.colorbar(mesh, ax=ax, pad=0.025, fraction=0.045,
                 label="Mean predicted susceptibility probability")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, facecolor="white")
    plt.close(fig)


def summarize_segments(model, values, pixel_segments):
    """Calculate means, medians and cell counts for one model."""
    records = []
    for segment in SEGMENTS:
        selected = values[pixel_segments == segment]
        records.append({
            "model": model,
            "segment": segment,
            "mean_prob": float(np.nanmean(selected)) if selected.size else np.nan,
            "median_prob": float(np.nanmedian(selected)) if selected.size else np.nan,
            "n_pixels": int(selected.size),
        })
    return records


def main(figure=None):
    check_inputs()
    order = model_order()

    buffer_gdf, rail, stations = load_vectors()
    buffer_geoms = [g.__geo_interface__ for g in buffer_gdf.geometry]

    line = centreline(rail)
    chainage = station_chainage(line, stations, station_code_field(stations))

    line_xy, line_chainages = densify(line)
    line_segments = np.array(
        [segment_of(c, chainage) for c in line_chainages], dtype=object
    )
    keep = np.array([s is not None for s in line_segments])
    tree = cKDTree(line_xy[keep])
    line_segments = line_segments[keep]

    print()
    records = []
    for model in order:
        coords, values = clipped_pixels(RASTERS[model], buffer_geoms)
        _, nearest = tree.query(coords, k=1)
        pixel_segments = line_segments[nearest]

        records.extend(summarize_segments(model, values, pixel_segments))

        print(f"  {model:<28} {len(coords):>8,} pixels")

    df = pd.DataFrame(records)
    df["rank"] = df.groupby("model")["mean_prob"].rank(ascending=False, method="min")

    empty = df[df.n_pixels == 0]
    if not empty.empty:
        print("\nWARNING: no pixels assigned to:")
        for _, row in empty.iterrows():
            print(f"  {row.model} / {row.segment}")

    STATS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    if figure is not None:
        plot_segment_means(df, Path(figure))

    print(f"\nStatistics -> {OUT_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", type=Path, help="Optional output path for the segment heatmap.")
    main(parser.parse_args().figure)

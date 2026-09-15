"""Interpolate station points from chainage in the QGIS Python Console.

Set the input paths below before running. The CSV must contain ``Loc`` and
``Distance (in meters)``. The railway must form one continuous line, with its
origin at Mettupalayam. Set REVERSE_LINE if its digitised direction is reversed.
The result is a temporary layer in the current QGIS project; export it to save it.
"""

import csv

from qgis.core import (
    QgsFeature,
    QgsField,
    QgsGeometry,
    QgsProject,
    QgsUnitTypes,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import QVariant

RAIL_SHAPEFILE = "/path/to/railway_line.shp"
STATION_CSV = "/path/to/station_chainage.csv"
REVERSE_LINE = False


def continuous_line(layer, reverse=False):
    """Merge all line features; reject disconnected or non-metric inputs."""
    if not layer.isValid():
        raise ValueError("Railway layer could not be loaded.")
    if layer.crs().mapUnits() != QgsUnitTypes.DistanceMeters:
        raise ValueError("Use a projected railway layer with metre units.")
    geometries = [feature.geometry() for feature in layer.getFeatures()]
    if not geometries:
        raise ValueError("Railway layer is empty.")
    line = QgsGeometry.unaryUnion(geometries).mergeLines()
    vertices = line.asPolyline()
    if len(vertices) < 2:
        raise ValueError("Railway features must form a single continuous line.")
    if reverse:
        line = QgsGeometry.fromPolylineXY(list(reversed(vertices)))
    return line


def read_stations(path, line_length):
    """Read the station names and validate chainages against line length."""
    with open(path, encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"Loc", "Distance (in meters)"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("CSV requires Loc and Distance (in meters) columns.")
        rows = [(row["Loc"], float(row["Distance (in meters)"])) for row in reader]
    if not rows:
        raise ValueError("Station CSV is empty.")
    for name, distance in rows:
        if not name.strip() or not 0 <= distance <= line_length:
            raise ValueError(f"Invalid station name or chainage: {name}, {distance}")
    return rows


def make_points(rail, rows, reverse=False):
    """Build one point per station using cumulative distance on the whole line."""
    line = continuous_line(rail, reverse)
    points = QgsVectorLayer(
        "Point?crs=" + rail.crs().authid(), "Railway stations", "memory"
    )
    provider = points.dataProvider()
    provider.addAttributes(
        [QgsField("Loc", QVariant.String), QgsField("Distance_m", QVariant.Double)]
    )
    points.updateFields()
    for name, distance in rows:
        point = line.interpolate(distance)
        if point.isNull() or point.isEmpty():
            raise ValueError(f"Cannot interpolate station {name} at {distance} m.")
        feature = QgsFeature(points.fields())
        feature.setGeometry(point)
        feature.setAttributes([name, distance])
        provider.addFeature(feature)
    points.updateExtents()
    return points


def main():
    rail = QgsVectorLayer(RAIL_SHAPEFILE, "Railway", "ogr")
    line = continuous_line(rail, REVERSE_LINE)
    rows = read_stations(STATION_CSV, line.length())
    points = make_points(rail, rows, REVERSE_LINE)
    QgsProject.instance().addMapLayer(points)
    print(f"Added {points.featureCount()} station points to the project.")


if __name__ == "__main__":
    main()

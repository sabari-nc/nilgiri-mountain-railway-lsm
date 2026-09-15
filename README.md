# Nilgiri Mountain Railway landslide susceptibility

Code and derived outputs for **“Corridor-Scale Landslide Susceptibility from a Long-Term Maintenance Inventory and Multi-Model Machine Learning, Nilgiri Mountain Railway, Southern India.”** The manuscript is submitted and unpublished.

**Authors:** Sabari Nathan Chellamuthu, Saravana Ganesh Manoharan, Chandra Hada, Iftekhar Ahmed and Ganapathy Pattukandan Ganapathy.

**GitHub:** https://github.com/sabari-nc/nilgiri-mountain-railway-lsm

**Zenodo DOI:** 10.5281/zenodo.21337329 — reserved; the archive is not yet published.

## Contents

| Folder | Contents |
|---|---|
| `code/` | MATLAB and Python training, SVM calibration and mapping, segment analysis, and a QGIS station utility |
| `models/` | Saved XGBoost model; MATLAB model objects are excluded |
| `susceptibility_maps/` | RF, BLR, SVM and XGBoost probability rasters; XGBoost ensemble standard deviation |
| `statistics/` | Test metrics, SVM calibration coefficients, feature importance and segment statistics |
| `vector_data/` | Corridor boundary, railway alignment and 12 stations |

RF denotes random forest; BLR, binary logistic regression; SVM, support vector machine. The models use 14 predictors. `manuscript_model_metrics.json` contains the rounded manuscript results. `svm_calibration.json` records the final calibration and unrounded SVM metrics. The importance files contain final-model RF out-of-bag permutation importance and XGBoost gain importance.

## Spatial data

All spatial files use WGS 84 / UTM zone 43N (EPSG:32643). Rasters are Float32 with 12.5 m cells, 2,318 columns and 1,088 rows. The analysis grid does not establish the native DEM resolution.

RF, BLR and SVM each contain 519,828 valid cells, with NaN elsewhere. XGBoost probability and uncertainty contain 482,190 valid cells, with NoData = -9999. The common valid-cell coverage is 482,190 cells. Segment statistics use each model's valid cells; comparisons over identical areas require a common mask.

SVM probabilities use logistic calibration. RF, BLR and XGBoost probabilities are uncalibrated model outputs. The uncertainty raster contains seven-member XGBoost ensemble standard deviations. Probability magnitudes should be interpreted with their differing calibration in mind.

Keep each shapefile's `.shp`, `.shx`, `.dbf` and `.prj` together. The 12 stations define 11 segments. Station fields `Station co` and `Kms` hold station code and chainage; `lat`/`lon` contain projected northing/easting in metres. The digitised line measures approximately 46.25 km; the nominal railway length is 45.9 km.

## Reproducing segment statistics

Use Python 3.11. From the repository root:

```bash
python -m pip install -r requirements.txt
python code/segment_susceptibility_stats.py
```

This regenerates `statistics/segment_susceptibility_stats.csv` from the supplied rasters and vectors. Columns contain model, segment, mean and median probability, valid-cell count and within-model rank. Add `--figure segment_wise.png` to save a heatmap.

## Model training

Training requires the restricted files `train_fp_final.csv`, `val_fp_final.csv`, `test_fp_final.csv` and `predictors_used_fp_final.txt`. Set `NMR_DATA_DIR` to their folder outside this repository. Training outputs remain under that folder and must not be uploaded without review.

MATLAB R2026a with the Statistics and Machine Learning Toolbox was used for the final MATLAB run. Run `train_ml_models.m` with `code/` on the MATLAB path. It saves model state and evaluation results under `NMR_DATA_DIR/model_runs/matlab/`. The Python implementation uses different model defaults and does not exactly reproduce MATLAB results.

`calibrate_svm.m` fits a logistic sigmoid to five-fold out-of-fold training scores and validation scores, using seed 42. Raw predictors are passed to the internally standardised SVM. Calibration folds are not spatially grouped; the outer test partition is excluded from calibration and threshold selection. The final validation threshold is 0.356 and test Brier score is 0.203. The supplied SVM raster and segment statistics supersede the earlier calibration.

`map_svm_probability.py` maps the saved MATLAB SVM and calibration coefficients using aligned predictor rasters. Run it with `--help` for input arguments. The saved model state and predictor rasters are not distributed.

XGBoost uses raw probabilities and version 3.1.2, as recorded in the saved model. Its 925 trees include early-stopping rounds; use `iteration_range=(0, 875)` for prediction. The training script selects Youden's J on 199 thresholds from 0.01 to 0.99, matching the original workflow. Archived XGBoost reports label average precision as `pr_auc` (0.597); the manuscript uses MATLAB `perfcurve` PR-AUC (0.596). These are different calculations, not different test predictions.

Complete retraining requires restricted inputs. Preprocessing, spatial-partition and ensemble-generation scripts are not included. The station utility requires QGIS and a station-chainage CSV.

## Data access, licence and citation

The Southern Railway landslide inventory, event locations, scanned registers, sample labels and training partitions are excluded. Original DEM, rainfall and predictor rasters are not supplied. Inventory access remains subject to Southern Railway's permission.

Included code and derived outputs are licensed under [CC BY 4.0](LICENSE.md). This licence does not cover excluded third-party data. Until publication, cite the title and authors above as an unpublished manuscript. Cite the Zenodo DOI once the archive is published.

**Contacts:** sabarinathan070@outlook.com; seismogans@yahoo.com.

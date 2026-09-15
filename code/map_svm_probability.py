"""Map a calibrated MATLAB SVM using aligned, restricted predictor rasters."""

import argparse
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import rasterio
from scipy.io import loadmat, savemat

FEATURES = [
    "F1_slope_deg", "F3_plan_curvature", "F4_profile_curvature",
    "F6_flow_accumulation", "F9_convergence_index", "F13_distance_to_drainage_m",
    "F14_drainage_density_km_km2", "aspect_sin", "aspect_cos",
    "lulc_10", "lulc_30", "lulc_40", "lulc_50", "lulc_60",
]


def read_aligned(path, reference):
    with rasterio.open(path) as src:
        if (src.shape, src.transform, src.crs) != (
            reference.shape, reference.transform, reference.crs
        ):
            raise ValueError(f"Raster grid differs from reference: {path}")
        return src.read(1, masked=True).astype("float64").filled(np.nan)


def predictor(name, factors, reference):
    if name.startswith("aspect_"):
        aspect = read_aligned(factors / "F2_aspect_deg.tif", reference)
        operation = {"aspect_sin": np.sin, "aspect_cos": np.cos}[name]
        return operation(np.deg2rad(aspect))
    if name.startswith("lulc_"):
        lulc = read_aligned(factors / "F10_LULC_aligned_corridor.tif", reference)
        values = (lulc == int(name.split("_")[1])).astype(float)
        values[~np.isfinite(lulc)] = np.nan
        return values
    return read_aligned(factors / f"{name}.tif", reference)


def matlab_string(value):
    return "'" + str(value).replace("'", "''") + "'"


def predict_matlab(model, matrix, executable):
    with tempfile.TemporaryDirectory(prefix="nmr-svm-") as folder:
        folder = Path(folder)
        inp, out = folder / "input.mat", folder / "output.mat"
        savemat(inp, {"X": matrix, "features": np.array(FEATURES, dtype=object)})
        script = f"""M=load({matlab_string(model)});
D=load({matlab_string(inp)});
assert(isequal(string(M.feature_names(:)),string(D.features(:))), 'Predictor order mismatch.');
assert(isequal(M.svmModel_raw.ClassNames(:),[0;1]), 'Expected classes 0 and 1.');
assert(strcmp(M.svmModel_raw.ScoreTransform,'none'), 'Expected raw decision scores.');
assert(numel(M.B_svm_recal)==2, 'Missing logistic calibration.');
p=zeros(size(D.X,1),1);
for first=1:40000:size(D.X,1)
    rows=first:min(first+39999,size(D.X,1));
    [~,scores]=predict(M.svmModel_raw,D.X(rows,:));
    p(rows)=glmval(M.B_svm_recal,scores(:,2),'logit');
end
save({matlab_string(out)},'p','-v7');
"""
        script_path = folder / "predict_grid.m"
        script_path.write_text(script)
        subprocess.run([executable, "-batch", f"run({matlab_string(script_path)})"], check=True)
        return loadmat(out)["p"].ravel()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--factors", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True,
                        help="Aligned raster defining the analysis mask, e.g. the BLR map.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matlab", default="matlab")
    args = parser.parse_args()
    with rasterio.open(args.reference) as ref:
        profile = ref.profile.copy()
        valid = np.isfinite(read_aligned(args.reference, ref))
        stack = np.stack([predictor(name, args.factors, ref) for name in FEATURES], axis=-1)
    valid &= np.all(np.isfinite(stack), axis=-1)
    if not valid.any():
        raise ValueError("No valid predictor cells.")
    probabilities = predict_matlab(args.model.resolve(), stack[valid], args.matlab)
    if not np.all(np.isfinite(probabilities) & (probabilities >= 0) & (probabilities <= 1)):
        raise ValueError("Invalid calibrated probabilities.")
    result = np.full(valid.shape, np.nan, dtype="float32")
    result[valid] = probabilities
    profile.update(dtype="float32", count=1, nodata=None, compress="deflate")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(args.output, "w", **profile) as dst:
        dst.write(result, 1)
    print(f"Wrote {valid.sum():,} valid cells to {args.output}")


if __name__ == "__main__":
    main()

% Train BLR, SVM (RBF + Platt calibration), and RF on the footprint-based
% landslide dataset for the Nilgiri Mountain Railway corridor.
% MATLAB R2026a
%
% Set NMR_DATA_DIR to the folder containing the restricted input files.
% The training/validation/test CSVs are not included in this repository;
% inventory access is subject to Southern Railway authorization.
%
% Predictor names are read from predictors_used_fp_final.txt so the
% exact feature set is locked at run time, preventing accidental inclusion
% of metadata columns. One-hot LULC columns are numeric. Positive-class
% probabilities are selected by ClassNames==1.

clear; clc;

%% ---------------- PATHS ----------------
ROOT = string(fileparts(fileparts(mfilename("fullpath"))));
[DATA, OUT] = local_data_paths(ROOT);

% All three partitions use the same screened predictor space,
% including aspect_sin and aspect_cos.
TR_CSV = fullfile(DATA, "train_fp_final.csv");
VA_CSV = fullfile(DATA, "val_fp_final.csv");
TE_CSV = fullfile(DATA, "test_fp_final.csv");

PRED_TXT = fullfile(DATA, "predictors_used_fp_final.txt");

OUT_MAT = fullfile(OUT, "matlab_model_results_fp.mat");

rng(42);

%% ---------------- LOAD TABLES ----------------
Ttr = readtable(TR_CSV);
Tva = readtable(VA_CSV);
Tte = readtable(TE_CSV);

%% ---------------- LABEL COLUMN (robust) ----------------
% Accept 'label' or 'y'
label_col = local_label_column(Ttr);
fprintf("Label column: %s\n", label_col);

ytr = double(Ttr.(label_col));
yva = double(Tva.(label_col));
yte = double(Tte.(label_col));

%% ---------------- LOCK PREDICTORS FROM predictors_used_fp_final.txt ----------------
predictor_names = local_predictors(PRED_TXT, {Ttr, Tva, Tte});
fprintf("Predictors (locked): %d\n", numel(predictor_names));
Xtr = double(Ttr{:, predictor_names});
Xva = double(Tva{:, predictor_names});
Xte = double(Tte{:, predictor_names});
feature_names = predictor_names(:);

%% ---------------- STANDARDIZE (train stats) ----------------
mu_tr  = mean(Xtr, 1, "omitnan");
sig_tr = std(Xtr, 0, 1, "omitnan");
sig_tr(sig_tr == 0) = 1;

Xtrz = (Xtr - mu_tr) ./ sig_tr;
Xvaz = (Xva - mu_tr) ./ sig_tr;
Xtez = (Xte - mu_tr) ./ sig_tr;

%% ---------------- METRIC HELPERS ----------------
auc_roc = @(y,p) perfcurve(y,p,1);
brier   = @(y,p) mean((p - y).^2);

conf_metrics = @(tp,fp,fn,tn) struct( ...
    "tp",tp,"fp",fp,"fn",fn,"tn",tn, ...
    "acc",(tp+tn)/(tp+tn+fp+fn), ...
    "balanced_acc",0.5*((tp/(tp+fn)) + (tn/(tn+fp))), ...
    "prec", tp / max(tp+fp,1), ...
    "rec",  tp / max(tp+fn,1), ...
    "f1",   (2*tp) / max(2*tp+fp+fn,1) );

best_f1_threshold = @(y,p) local_best_f1_threshold(y,p);

%% =======================================================================
% BLR — L1-regularised logistic regression
% Lambda selected by 5-fold CV on the training partition (lassoglm,
% Alpha=1.0, binomial). Applied to standardised predictors.
% =======================================================================
[B,FitInfo] = lassoglm(Xtrz, ytr, "binomial", "Alpha", 1.0, "CV", 5, "Standardize", false);
idxMin = FitInfo.IndexMinDeviance;
beta_lasso = [FitInfo.Intercept(idxMin); B(:,idxMin)];

p_blr_va = local_sigmoid([ones(size(Xvaz,1),1) Xvaz] * beta_lasso);
p_blr_te = local_sigmoid([ones(size(Xtez,1),1) Xtez] * beta_lasso);

[~,~,~,auc_blr_va] = auc_roc(yva, p_blr_va);
[~,~,~,auc_blr_te] = auc_roc(yte, p_blr_te);

[~,~,~,pr_blr_va] = perfcurve(yva, p_blr_va, 1, "xCrit","reca", "yCrit","prec");
[~,~,~,pr_blr_te] = perfcurve(yte, p_blr_te, 1, "xCrit","reca", "yCrit","prec");

thr_blr = best_f1_threshold(yva, p_blr_va);
yhat_blr = double(p_blr_te >= thr_blr);
[tp,fp,fn,tn] = local_confusion(yte, yhat_blr);

blr_metrics = conf_metrics(tp,fp,fn,tn);
blr_metrics.threshold = thr_blr;
blr_metrics.brier = brier(yte, p_blr_te);

%% =======================================================================
% SVM — RBF kernel with Platt calibration
% Store the raw decision-score model and its logistic calibration coefficients.
% =======================================================================
svmRaw = fitcsvm(Xtr, ytr, ...
    "KernelFunction","rbf", ...
    "Standardize",true, ...           % internal z-score using TRAIN stats
    "ClassNames",[0 1], ...
    "KernelScale","auto");

svmModel_raw = svmRaw;
[B_svm_recal, p_svm_va, p_svm_te, svmModel_cal] = local_calibrate_svm(svmModel_raw, Xva, yva, Xte);

[~,~,~,auc_svm_va] = auc_roc(yva, p_svm_va);
[~,~,~,auc_svm_te] = auc_roc(yte, p_svm_te);

[~,~,~,pr_svm_va] = perfcurve(yva, p_svm_va, 1, "xCrit","reca", "yCrit","prec");
[~,~,~,pr_svm_te] = perfcurve(yte, p_svm_te, 1, "xCrit","reca", "yCrit","prec");

thr_svm = best_f1_threshold(yva, p_svm_va);
yhat_svm = double(p_svm_te >= thr_svm);
[tp,fp,fn,tn] = local_confusion(yte, yhat_svm);

svm_metrics = conf_metrics(tp,fp,fn,tn);
svm_metrics.threshold = thr_svm;
svm_metrics.brier = brier(yte, p_svm_te);

%% =======================================================================
% RF — TreeBagger tuned by validation ROC-AUC
% Uses raw (unstandardised) predictors. MinLeafSize is chosen from six
% candidates using validation-set AUC. OOB importance is recorded for
% later analysis. The supplied one-hot LULC predictors are numeric.
% =======================================================================
leaf_grid = [1 2 5 10 20 50];

bestLeaf   = leaf_grid(1);
bestValAUC = -inf;
bestOOB    = NaN;
rf_best    = [];

cat_idx = find(feature_names == "F10_LULC_aligned_corridor");

for L = leaf_grid

    rf_tmp = TreeBagger(800, Xtr, ytr, ...
        "Method","classification", ...
        "MinLeafSize", L, ...
        "OOBPrediction","on", ...
        "OOBPredictorImportance","on", ...
        "PredictorNames", cellstr(feature_names), ...
        "CategoricalPredictors", cat_idx);

    % --- VAL probabilities for class==1 ---
    [~, score_va_tmp] = predict(rf_tmp, Xva);
    p_va_tmp = local_pick_pos_score(rf_tmp.ClassNames, score_va_tmp);

    % VAL ROC-AUC
    [~,~,~,auc_val_tmp] = perfcurve(yva, p_va_tmp, 1);

    % OOB AUC (for reporting only)
    [~, oobScore] = oobPredict(rf_tmp);
    p_oob_tmp = local_pick_pos_score(rf_tmp.ClassNames, oobScore);
    [~,~,~,auc_oob_tmp] = perfcurve(ytr, p_oob_tmp, 1);

    if auc_val_tmp > bestValAUC
        bestValAUC = auc_val_tmp;
        bestLeaf   = L;
        bestOOB    = auc_oob_tmp;
        rf_best    = rf_tmp;
    end
end

rf = rf_best;

fprintf("\nRF tuned by VAL AUC. BestLeaf=%d | VAL-AUC=%.5f | OOB-AUC=%.5f\n", bestLeaf, bestValAUC, bestOOB);

% --- VAL/TEST probabilities for chosen RF ---
[~, score_va] = predict(rf, Xva);
[~, score_te] = predict(rf, Xte);

p_rf_va = local_pick_pos_score(rf.ClassNames, score_va);
p_rf_te = local_pick_pos_score(rf.ClassNames, score_te);

% --- RF AUC/PR ---
[~,~,~,auc_rf_va] = perfcurve(yva, p_rf_va, 1);
[~,~,~,auc_rf_te] = perfcurve(yte, p_rf_te, 1);

[~,~,~,pr_rf_va]  = perfcurve(yva, p_rf_va, 1, "xCrit","reca", "yCrit","prec");
[~,~,~,pr_rf_te]  = perfcurve(yte, p_rf_te, 1, "xCrit","reca", "yCrit","prec");

% --- Threshold on VAL (max F1), evaluate on TEST ---
thr_rf  = best_f1_threshold(yva, p_rf_va);
yhat_rf = double(p_rf_te >= thr_rf);
[tp,fp,fn,tn] = local_confusion(yte, yhat_rf);

rf_metrics = conf_metrics(tp,fp,fn,tn);
rf_metrics.threshold = thr_rf;
rf_metrics.brier = mean((p_rf_te - yte).^2);

%% ---------------- PRINT SUMMARY ----------------
fprintf("\nBLR  ROC-AUC val/test: %.5f / %.5f\n", auc_blr_va, auc_blr_te);
fprintf("BLR  PR-AUC  val/test: %.5f / %.5f\n", pr_blr_va, pr_blr_te);
local_print_metrics(blr_metrics);

fprintf("\nSVM (calibrated)  ROC-AUC val/test: %.5f / %.5f\n", auc_svm_va, auc_svm_te);
fprintf("SVM (calibrated)  PR-AUC  val/test: %.5f / %.5f\n", pr_svm_va, pr_svm_te);
local_print_metrics(svm_metrics);

fprintf("\nRF (VAL-tuned) BestLeaf=%d | VAL-AUC=%.5f | OOB-AUC=%.5f\n", bestLeaf, bestValAUC, bestOOB);
fprintf("RF            ROC-AUC val/test: %.5f / %.5f\n", auc_rf_va, auc_rf_te);
fprintf("RF            PR-AUC  val/test: %.5f / %.5f\n", pr_rf_va, pr_rf_te);
local_print_metrics(rf_metrics);

%% =======================================================================
% SAVE — models, predictions, thresholds, and metrics
% =======================================================================
S = struct();

% Predictor matrices
S.Xtr = Xtr; S.ytr = ytr;
S.Xva = Xva; S.yva = yva;
S.Xte = Xte; S.yte = yte;

% Standardized matrices for BLR; the SVM receives raw predictors.
S.Xtrz = Xtrz; S.Xvaz = Xvaz; S.Xtez = Xtez;

% Locked feature list; must match X column order exactly
S.feature_names   = feature_names;
S.predictor_names = feature_names;  % alias used by permutation-importance scripts

% Standardization stats (needed by mapping scripts)
S.mu_tr  = mu_tr;
S.sig_tr = sig_tr;

% ---- Models ----
S.blr_beta_lasso = beta_lasso;   % coefficients in standardized space

% ---- SVM ----
S.svmModel_raw  = svmModel_raw;   % raw fitcsvm (decision scores)
S.B_svm_recal = B_svm_recal; % logistic intercept and slope
S.svmModel_cal = svmModel_cal;
S.svmCalib = svmModel_cal;

S.svmModel = svmModel_raw;


S.rf             = rf;           % tuned TreeBagger (OOBPredictorImportance ON)

% ---- Probabilities ----
S.p_blr_va = p_blr_va; S.p_blr_te = p_blr_te;
S.p_svm_va = p_svm_va; S.p_svm_te = p_svm_te;
S.p_rf_va  = p_rf_va;  S.p_rf_te  = p_rf_te;

% ---- Thresholds ----
S.thr_blr = thr_blr;
S.thr_svm = thr_svm;
S.thr_rf  = thr_rf;

% ---- AUC summaries ----
S.aucs = struct( ...
    "blr_val_roc", auc_blr_va, "blr_test_roc", auc_blr_te, "blr_val_pr", pr_blr_va, "blr_test_pr", pr_blr_te, ...
    "svm_val_roc", auc_svm_va, "svm_test_roc", auc_svm_te, "svm_val_pr", pr_svm_va, "svm_test_pr", pr_svm_te, ...
    "rf_val_roc",  auc_rf_va,  "rf_test_roc",  auc_rf_te,  "rf_val_pr",  pr_rf_va,  "rf_test_pr",  pr_rf_te, ...
    "rf_oob_auc",  bestOOB,    "rf_best_leaf", bestLeaf,   "rf_best_val_auc", bestValAUC ...
);

% ---- Confusion-matrix metrics ----
S.metrics = struct( ...
    "blr", blr_metrics, ...
    "svm", svm_metrics, ...
    "rf",  rf_metrics ...
);

save(OUT_MAT, "-struct", "S", "-v7.3");
fprintf("\nSaved: %s\n\n", OUT_MAT);

%% ======================= LOCAL FUNCTIONS ==========================
function [DATA, OUT] = local_data_paths(ROOT)
    DATA = string(getenv("NMR_DATA_DIR"));
    assert(strlength(DATA) > 0 && isfolder(DATA), ...
        "Set NMR_DATA_DIR to the restricted input folder.");
    data_path = string(java.io.File(char(DATA)).getCanonicalPath());
    repo_path = string(java.io.File(char(ROOT)).getCanonicalPath());
    assert(~strcmpi(data_path, repo_path) && ...
        ~startsWith(lower(data_path), lower(repo_path + filesep)), ...
        "Keep restricted inputs and outputs outside this repository.");
    OUT = fullfile(DATA, "model_runs", "matlab");
    if ~isfolder(OUT)
        mkdir(OUT);
    end

end

function label_col = local_label_column(Ttr)
    if any(strcmpi(Ttr.Properties.VariableNames, "label"))
        label_col = string(Ttr.Properties.VariableNames{ ...
            find(strcmpi(Ttr.Properties.VariableNames, "label"), 1)});
    elseif any(strcmpi(Ttr.Properties.VariableNames, "y"))
        label_col = string(Ttr.Properties.VariableNames{ ...
            find(strcmpi(Ttr.Properties.VariableNames, "y"), 1)});
    else
        error("Could not find label column. Expected 'label' or 'y'.");
    end
end

function pred_list = local_predictors(path, tables)
    assert(isfile(path), "Missing predictors_used_fp_final.txt: %s", path);
    pred_list = string(strtrim(splitlines(fileread(path))));
    pred_list = unique(pred_list(pred_list ~= ""), "stable");
    split_names = ["Train", "Validation", "Test"];
    for i = 1:numel(tables)
        local_check_predictors(tables{i}, pred_list, split_names(i));
    end
    if ~all(ismember(["aspect_sin", "aspect_cos"], pred_list))
        error("Predictor set must include aspect_sin and aspect_cos.");
    end
    if any(pred_list == "F2_aspect_deg")
        error("Raw aspect (F2_aspect_deg) must not be in the predictor set.");
    end
end

function local_check_predictors(T, pred_list, split_name)
    missing = pred_list(~ismember(pred_list, string(T.Properties.VariableNames)));
    if ~isempty(missing)
        error("%s predictors missing: %s", split_name, strjoin(missing, ", "));
    end
    for i = 1:numel(pred_list)
        values = T.(pred_list(i));
        if ~(isnumeric(values) || islogical(values))
            error("Non-numeric %s predictor: %s", split_name, pred_list(i));
        end
    end
end

function p = local_pick_pos_score(classNames, scoreMat)
    % Return the column in scoreMat that corresponds to class==1.
    % Works for numeric or cellstring ClassNames.
    if isempty(scoreMat)
        error("Empty score matrix.");
    end

    cn = classNames;
    if iscell(cn)
        cn_num = str2double(string(cn));
    else
        cn_num = double(cn);
    end

    k1 = find(cn_num == 1, 1);
    if isempty(k1)
        error("Positive class 1 is missing from ClassNames.");
    end

    p = scoreMat(:, k1);
end

function p = local_sigmoid(z)
    p = 1 ./ (1 + exp(-z));
end

function [tp,fp,fn,tn] = local_confusion(ytrue, yhat)
    ytrue = double(ytrue(:));
    yhat  = double(yhat(:));
    tp = sum(ytrue==1 & yhat==1);
    fp = sum(ytrue==0 & yhat==1);
    fn = sum(ytrue==1 & yhat==0);
    tn = sum(ytrue==0 & yhat==0);
end

function thr = local_best_f1_threshold(y, p)
    grid = linspace(0.001, 0.999, 999);
    bestF1 = -inf;
    thr = 0.5;
    for t = grid
        yhat = double(p >= t);
        [tp,fp,fn,~] = local_confusion(y, yhat);
        denom = (2*tp + fp + fn);
        f1 = (2*tp) / max(denom,1);
        if f1 > bestF1
            bestF1 = f1;
            thr = t;
        end
    end
end

function local_print_metrics(M)
    fprintf("             thr: %.4f\n", M.threshold);
    fprintf("              tp: %d\n", M.tp);
    fprintf("              fp: %d\n", M.fp);
    fprintf("              fn: %d\n", M.fn);
    fprintf("              tn: %d\n", M.tn);
    fprintf("             acc: %.4f\n", M.acc);
    fprintf("    balanced_acc: %.4f\n", M.balanced_acc);
    fprintf("            prec: %.4f\n", M.prec);
    fprintf("             rec: %.4f\n", M.rec);
    fprintf("              f1: %.4f\n", M.f1);
    fprintf("           brier: %.4f\n", M.brier);
end

function [B, p_validation, p_test, calibrated] = local_calibrate_svm(model, X_validation, y_validation, X_test)
% Fit a logistic calibration using five-fold training scores and validation.
% Pass raw predictors; the fitted SVM applies its own standardization.
    assert(isequal(model.ClassNames(:), [0; 1]), 'Expected classes 0 and 1.');
    assert(strcmp(model.ScoreTransform, 'none'), 'Supply the uncalibrated SVM.');
    previous_rng = rng;
    restore_rng = onCleanup(@() rng(previous_rng));
    rng(42, 'twister');
    partitioned = crossval(model, 'KFold', 5);
    [~, training_scores] = kfoldPredict(partitioned);
    [~, validation_scores] = predict(model, X_validation);
    B = glmfit([training_scores(:, 2); validation_scores(:, 2)], ...
        [double(model.Y(:)); double(y_validation(:))], 'binomial', 'link', 'logit');
    p_validation = glmval(B, validation_scores(:, 2), 'logit');
    [~, test_scores] = predict(model, X_test);
    p_test = glmval(B, test_scores(:, 2), 'logit');
    calibrated = model;
    calibrated.ScoreTransform = @(scores) [ ...
        1 - glmval(B, scores(:, 2), 'logit'), glmval(B, scores(:, 2), 'logit')];
end

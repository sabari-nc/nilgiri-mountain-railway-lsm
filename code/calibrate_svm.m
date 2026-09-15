function [B, p_validation, p_test, calibrated] = calibrate_svm(model, X_validation, y_validation, X_test)
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

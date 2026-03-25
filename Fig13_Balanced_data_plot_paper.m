%% VesselMNIST3D: LSRTR vs MUON with Early Stopping
clear; clc; close all;
rng(1200);

%% Configuration
load('vesselmnist3d_balanced_v2.mat');

mm = [28 28 28];
K  = length(mm);
m  = prod(mm);

n_train = size(X, 1);
n_test  = size(Xt, 1);

y_train = double(y_train(:));
y_test  = double(y_test(:));

%% LSR model parameters
rr = [5 5 5];
r  = prod(rr);
S  = 3;

num_trials = 10;
max_iter = 50;
alpha      = 0.5;
alpha_muon = 0.07;
beta       = 0.1;
lam        = 1e-2;

% Early stopping parameters
patience_lsrtr = 10;
patience_muon = 10;
min_delta = 0;
min_iters_lsrtr = 20;
min_iters_muon = 15;

%% Storage for all trials
all_loss_lsrtr = nan(max_iter, num_trials);
all_loss_muon = nan(max_iter, num_trials);
all_erry_lsrtr = nan(max_iter, num_trials);
all_erry_muon = nan(max_iter, num_trials);


all_metrics_history_lsrtr = nan(max_iter, num_trials, 5); 
all_metrics_history_muon = nan(max_iter, num_trials, 5);

all_time_per_iter_lsrtr = nan(max_iter, num_trials);
all_time_per_iter_muon = nan(max_iter, num_trials);
% ==============================================

all_metrics_lsrtr = zeros(num_trials, 5);
all_metrics_muon = zeros(num_trials, 5);

all_time_lsrtr = zeros(num_trials, 1);
all_time_muon = zeros(num_trials, 1);

actual_iters_lsrtr = zeros(num_trials, 1);
actual_iters_muon = zeros(num_trials, 1);

%% Run multiple trials
fprintf('Starting %d trials with early stopping...\n', num_trials);

for trial = 1:num_trials
    fprintf('\n========== Trial %d/%d ==========\n', trial, num_trials);
    
    % Random initialization
    G_0 = randn(rr) / sqrt(r);
    
    B_0 = cell(K, S);
    for s = 1:S
        for k = 1:K
            [Q,R] = qr(randn(mm(k), rr(k)), 0);
            Q = Q * diag(sign(diag(R) + (diag(R)==0)));
            B_0{k,s} = Q(:,1:rr(k));
        end
    end
    
    M_0 = cell(K, S);
    for s = 1:S
        for k = 1:K
            M_0{k,s} = zeros(mm(k), rr(k)); 
        end
    end
    
    loss_history = nan(max_iter, 1);
    loss_history_muon = nan(max_iter, 1);
    erry_history = nan(max_iter, 1);
    erry_history_muon = nan(max_iter, 1);
    
    per_iter_lsrtr = zeros(max_iter, 1);
    per_iter_muon = zeros(max_iter, 1);
    
    B_current = B_0; 
    B_current_muon = B_0; 
    M_current_muon = M_0;
    G_current = G_0;
    G_current_muon = G_0;
    
    % Early stopping variables
    best_erry_lsrtr = inf;
    best_erry_muon = inf;
    wait_lsrtr = 0;
    wait_muon = 0;
    stopped_lsrtr = false;
    stopped_muon = false;
    
    % Store best models
    best_B_lsrtr = B_current;
    best_G_lsrtr = G_current;
    best_B_muon = B_current_muon;
    best_G_muon = G_current_muon;
    
    %% Training loop
    for iter = 1:max_iter
        if mod(iter, 10) == 0
            lsrtr_status = 'running';
            if stopped_lsrtr
                lsrtr_status = 'STOPPED';
            end
            muon_status = 'running';
            if stopped_muon
                muon_status = 'STOPPED';
            end
            fprintf('  iter = %d [LSRTR: %s, MUON: %s]\n', iter, ...
                    lsrtr_status, muon_status);
        end
        
        %% ========== LSRTR: one iteration (if not stopped) ========== 
        if ~stopped_lsrtr
            t0 = tic;
            
            vec_G_current = G_current(:);
            
            % Update factor matrices
            for s_prime = 1:S
                for k_prime = 1:K
                    tilde_B_current = zeros(m,r);
                    for s = 1:S
                        Btemp = B_current{K,s};
                        for k = (K-1):-1:1
                            Btemp = kron(Btemp, B_current{k,s});
                        end
                        tilde_B_current = tilde_B_current + Btemp;
                    end
                    vec_B_current = tilde_B_current * vec_G_current;
                    
                    tilde_B_s_prime = zeros(m,r);
                    Btemp = B_current{K,s_prime};
                    for k = (K-1):-1:1
                        Btemp = kron(Btemp, B_current{k,s_prime});
                    end
                    tilde_B_s_prime = tilde_B_s_prime + Btemp;
                    vec_B_s_prime = tilde_B_s_prime * vec_G_current;
                    B_s_prime = reshape(vec_B_s_prime, mm);
                    B_s_prime_unfold = mode_k_unfold(B_s_prime, k_prime);
                    
                    G_unfold = mode_k_unfold(G_current, k_prime);
                    
                    order = [k_prime, K:-1:k_prime+1, k_prime-1:-1:1];
                    order_rest = order(2:end);
                    Btemp = [];
                    for idx = 1:numel(order_rest)
                        t = order_rest(idx);
                        if isempty(Btemp)
                            Btemp = B_current{t, s_prime};
                        else
                            Btemp = kron(Btemp, B_current{t, s_prime});
                        end
                    end
                    tilde_B_s_prime_neqkp = Btemp;
                    
                    M_kp_sp = tilde_B_s_prime_neqkp * G_unfold';
                    
                    Omega_kp_sp = zeros(mm(k_prime)*rr(k_prime), n_train);
                    for i = 1:n_train
                        Xi = reshape(X(i,:)', mm);
                        Xi_unfold = mode_k_unfold(Xi, k_prime);
                        Otemp = Xi_unfold * M_kp_sp;
                        Omega_kp_sp(:,i) = Otemp(:);
                    end
                    
                    logits_current = X * vec_B_current;
                    prob_current = 1./(1 + exp(-logits_current));
                    residual = prob_current - y_train;
                    
                    vec_grad_B = Omega_kp_sp * residual / n_train;
                    grad_B = reshape(vec_grad_B, mm(k_prime), rr(k_prime));
                    
                    B_temp = B_current{k_prime, s_prime} - alpha * grad_B;
                    [Q,~] = qr(B_temp, 0);
                    B_current{k_prime, s_prime} = Q;
                end
            end
            
            % Update core tensor G
            tilde_B_current = zeros(m,r);
            for s = 1:S
                Btemp = B_current{K,s};
                for k = (K-1):-1:1
                    Btemp = kron(Btemp, B_current{k,s});
                end
                tilde_B_current = tilde_B_current + Btemp;
            end
            vec_B_current = tilde_B_current*vec_G_current;
            
            XB = X * tilde_B_current;
            logits_current = X * vec_B_current;
            prob_current = 1./(1 + exp(-logits_current));
            residual = prob_current - y_train;
            
            vec_grad_G = (XB)'*residual/n_train;
            vec_G_current = vec_G_current - alpha * vec_grad_G;
            G_current = reshape(vec_G_current, rr);
            
            per_iter_lsrtr(iter) = toc(t0);
            
         
            all_time_per_iter_lsrtr(iter, trial) = per_iter_lsrtr(iter);
            
            % Compute metrics
            vec_B_current = tilde_B_current * vec_G_current;
            eta_train = X * vec_B_current;
            loss_history(iter) = mean(-y_train.*eta_train + log(1 + exp(eta_train)));
            
            p_test = 1./(1 + exp(-(Xt * vec_B_current)));
            y_pred = double(p_test > 0.5);
            erry_history(iter) = mean(abs(y_pred - y_test));
            
          
            metrics_iter = compute_metrics(y_test, y_pred, p_test);
            all_metrics_history_lsrtr(iter, trial, :) = [metrics_iter.Sensitivity, ...
                metrics_iter.Specificity, metrics_iter.F1, metrics_iter.AUC, metrics_iter.Accuracy];
          
            
            % Early stopping check for LSRTR
            if iter >= min_iters_lsrtr
                if erry_history(iter) < best_erry_lsrtr - min_delta
                    best_erry_lsrtr = erry_history(iter);
                    best_B_lsrtr = B_current;
                    best_G_lsrtr = G_current;
                    wait_lsrtr = 0;
                else
                    wait_lsrtr = wait_lsrtr + 1;
                    if wait_lsrtr >= patience_lsrtr
                        stopped_lsrtr = true;
                        actual_iters_lsrtr(trial) = iter;
                        fprintf('  LSRTR early stopped at iter %d (best error: %.4f)\n', ...
                                iter, best_erry_lsrtr);
                        % Restore best model
                        B_current = best_B_lsrtr;
                        G_current = best_G_lsrtr;
                    end
                end
            end
        else
            % NAN
            loss_history(iter) = NaN;
            erry_history(iter) = NaN;
        end
        
        %% ========== MUON: one iteration (if not stopped) ========== 
        if ~stopped_muon
            t1 = tic;
            
            vec_G_current_muon = G_current_muon(:);
            
            for s_prime = 1:S
                for k_prime = 1:K
                    tilde_B_current_muon = zeros(m,r);
                    for s = 1:S
                        Btemp = B_current_muon{K,s};
                        for k = (K-1):-1:1
                            Btemp = kron(Btemp, B_current_muon{k,s});
                        end
                        tilde_B_current_muon = tilde_B_current_muon + Btemp;
                    end
                    vec_B_current_muon = tilde_B_current_muon * vec_G_current_muon;          
                    
                    tilde_B_s_prime_muon = zeros(m,r);
                    Btemp = B_current_muon{K,s_prime};
                    for k = (K-1):-1:1
                        Btemp = kron(Btemp, B_current_muon{k,s_prime});
                    end
                    tilde_B_s_prime_muon = tilde_B_s_prime_muon + Btemp;
                    
                    vec_B_s_prime_muon = tilde_B_s_prime_muon * vec_G_current_muon;
                    B_s_prime_muon = reshape(vec_B_s_prime_muon, mm);
                    B_s_prime_muon_unfold = mode_k_unfold(B_s_prime_muon, k_prime);
                    
                    G_muon_unfold = mode_k_unfold(G_current_muon, k_prime);
                    
                    order = [k_prime, K:-1:k_prime+1, k_prime-1:-1:1];
                    order_rest = order(2:end);
                    
                    Btemp = [];
                    for idx = 1:numel(order_rest)
                        t = order_rest(idx);
                        if isempty(Btemp)
                            Btemp = B_current_muon{t, s_prime};
                        else
                            Btemp = kron(Btemp, B_current_muon{t, s_prime});
                        end
                    end
                    tilde_B_s_prime_neqkp_muon = Btemp;
                    
                    M_kp_sp_muon = tilde_B_s_prime_neqkp_muon * G_muon_unfold';
                    
                    Omega_kp_sp_muon = zeros(mm(k_prime)*rr(k_prime), n_train);
                    for i = 1:n_train
                        Xi = reshape(X(i,:)', mm);
                        Xi_unfold = mode_k_unfold(Xi, k_prime);
                        Otemp = Xi_unfold * M_kp_sp_muon;
                        Omega_kp_sp_muon(:,i) = Otemp(:);
                    end
                    
                    logits_muon = X * vec_B_current_muon;
                    prob_muon = 1./(1 + exp(-logits_muon));
                    residual_muon = prob_muon - y_train;
                    
                    vec_grad_B_muon = Omega_kp_sp_muon * residual_muon / n_train;
                    grad_B_muon = reshape(vec_grad_B_muon, mm(k_prime), rr(k_prime));
                    
                    M_current_muon{k_prime, s_prime} = beta*M_current_muon{k_prime, s_prime} + grad_B_muon;
                    
                    eps_reg = 1e-4;
                    Mcm = M_current_muon{k_prime, s_prime};
                    Acm = Mcm' * Mcm + eps_reg * eye(size(Mcm,2));
                    [Ucm,Scm] = eig(Acm);
                    A_invhalf = Ucm * diag(1./sqrt(diag(Scm))) * Ucm';
                    Qcm = Mcm * A_invhalf;
                    
                    B_current_muon{k_prime, s_prime} = B_current_muon{k_prime, s_prime} - ...
                        alpha_muon*(Qcm + lam*B_current_muon{k_prime, s_prime});
                end
            end
            
            % Update core tensor G
            tilde_B_current_muon = zeros(m,r);
            for s = 1:S
                Btemp = B_current_muon{K,s};
                for k = (K-1):-1:1
                    Btemp = kron(Btemp, B_current_muon{k,s});
                end
                tilde_B_current_muon = tilde_B_current_muon + Btemp;
            end
            vec_B_current_muon = tilde_B_current_muon * vec_G_current_muon;
            
            XB_muon = X * tilde_B_current_muon;
            logits_muon = X * vec_B_current_muon;
            prob_muon = 1./(1 + exp(-logits_muon));
            residual_muon = prob_muon - y_train;
            
            vec_grad_G_muon = (XB_muon)'*residual_muon/n_train;
            vec_G_current_muon = vec_G_current_muon - alpha * vec_grad_G_muon;
            G_current_muon = reshape(vec_G_current_muon, rr);
            
            per_iter_muon(iter) = toc(t1);
            
           
            all_time_per_iter_muon(iter, trial) = per_iter_muon(iter);
            
            % Compute metrics
            vec_B_current_muon = tilde_B_current_muon * vec_G_current_muon;
            eta_train_muon = X * vec_B_current_muon;
            loss_history_muon(iter) = mean(-y_train.*eta_train_muon + log(1 + exp(eta_train_muon)));
            
            p_test_muon = 1./(1 + exp(-(Xt * vec_B_current_muon)));
            y_pred_muon = double(p_test_muon > 0.5);
            erry_history_muon(iter) = mean(abs(y_pred_muon - y_test));
            
         
            metrics_iter_muon = compute_metrics(y_test, y_pred_muon, p_test_muon);
            all_metrics_history_muon(iter, trial, :) = [metrics_iter_muon.Sensitivity, ...
                metrics_iter_muon.Specificity, metrics_iter_muon.F1, metrics_iter_muon.AUC, metrics_iter_muon.Accuracy];
            
            
            % Early stopping check for MUON
            if iter >= min_iters_muon
                if erry_history_muon(iter) < best_erry_muon - min_delta
                    best_erry_muon = erry_history_muon(iter);
                    best_B_muon = B_current_muon;
                    best_G_muon = G_current_muon;
                    wait_muon = 0;
                else
                    wait_muon = wait_muon + 1;
                    if wait_muon >= patience_muon
                        stopped_muon = true;
                        actual_iters_muon(trial) = iter;
                        fprintf('  MUON early stopped at iter %d (best error: %.4f)\n', ...
                                iter, best_erry_muon);
                        % Restore best model
                        B_current_muon = best_B_muon;
                        G_current_muon = best_G_muon;
                    end
                end
            end
        else
            % NaN
            loss_history_muon(iter) = NaN;
            erry_history_muon(iter) = NaN;
        end
        
        % If both stopped, break the loop
        if stopped_lsrtr && stopped_muon
            fprintf('  Both algorithms stopped, ending iteration loop.\n');
            break;
        end
    end
    
    % If not stopped by early stopping, set actual iters to max
    if actual_iters_lsrtr(trial) == 0
        actual_iters_lsrtr(trial) = max_iter;
    end
    if actual_iters_muon(trial) == 0
        actual_iters_muon(trial) = max_iter;
    end
    
    % Store results for this trial
    all_loss_lsrtr(:, trial) = loss_history;
    all_loss_muon(:, trial) = loss_history_muon;
    all_erry_lsrtr(:, trial) = erry_history;
    all_erry_muon(:, trial) = erry_history_muon;
    
    % Store total running time (only up to convergence)
    all_time_lsrtr(trial) = sum(per_iter_lsrtr(1:actual_iters_lsrtr(trial)));
    all_time_muon(trial) = sum(per_iter_muon(1:actual_iters_muon(trial)));
    
    % Final evaluation using best models
    p_test_L = 1./(1 + exp(-(Xt * vec_B_current)));
    y_pred_L = double(p_test_L > 0.5);
    
    p_test_M = 1./(1 + exp(-(Xt * vec_B_current_muon)));
    y_pred_M = double(p_test_M > 0.5);
    
    metrics_L = compute_metrics(y_test, y_pred_L, p_test_L);
    metrics_M = compute_metrics(y_test, y_pred_M, p_test_M);
    
    all_metrics_lsrtr(trial, :) = [metrics_L.Sensitivity, metrics_L.Specificity, ...
                                    metrics_L.F1, metrics_L.AUC, metrics_L.Accuracy];
    all_metrics_muon(trial, :) = [metrics_M.Sensitivity, metrics_M.Specificity, ...
                                   metrics_M.F1, metrics_M.AUC, metrics_M.Accuracy];
    
    fprintf('  Trial %d: LSRTR Acc=%.4f (%d iters, %.2fs), MUON Acc=%.4f (%d iters, %.2fs)\n', ...
        trial, metrics_L.Accuracy, actual_iters_lsrtr(trial), all_time_lsrtr(trial), ...
        metrics_M.Accuracy, actual_iters_muon(trial), all_time_muon(trial));
end

%% Compute statistics
mean_metrics_lsrtr = mean(all_metrics_lsrtr, 1);
mean_metrics_muon = mean(all_metrics_muon, 1);

mean_erry_lsrtr = mean(all_erry_lsrtr, 2, 'omitnan');
mean_erry_muon  = mean(all_erry_muon,  2, 'omitnan');

mean_time_lsrtr = mean(all_time_lsrtr);
mean_time_muon = mean(all_time_muon);

mean_iters_lsrtr = mean(actual_iters_lsrtr);
mean_iters_muon = mean(actual_iters_muon);


%% Display results
fprintf('\n\n========================================\n');
fprintf('FINAL RESULTS (Average over %d trials)\n', num_trials);
fprintf('========================================\n\n');

fprintf('Metric               LSRTR           MUON\n');
fprintf('---------------------------------------------\n');
fprintf('%-20s %-15.4f %-15.4f\n', 'Sensitivity', mean_metrics_lsrtr(1), mean_metrics_muon(1));
fprintf('%-20s %-15.4f %-15.4f\n', 'Specificity', mean_metrics_lsrtr(2), mean_metrics_muon(2));
fprintf('%-20s %-15.4f %-15.4f\n', 'F1 Score', mean_metrics_lsrtr(3), mean_metrics_muon(3));
fprintf('%-20s %-15.4f %-15.4f\n', 'AUC', mean_metrics_lsrtr(4), mean_metrics_muon(4));
fprintf('%-20s %-15.4f %-15.4f\n', 'Accuracy', mean_metrics_lsrtr(5), mean_metrics_muon(5));
fprintf('%-20s %-15.1f %-15.1f\n', 'Avg Iterations', mean_iters_lsrtr, mean_iters_muon);
fprintf('%-20s %-15.2f %-15.2f\n', 'Running Time (s)', mean_time_lsrtr, mean_time_muon);
fprintf('---------------------------------------------\n');






%% Helper functions
function metrics = compute_metrics(y_true, y_hat, scores)
    TP = sum((y_hat==1) & (y_true==1));
    TN = sum((y_hat==0) & (y_true==0));
    FP = sum((y_hat==1) & (y_true==0));
    FN = sum((y_hat==0) & (y_true==1));
    
    metrics = struct( ...
        'TP', TP, 'TN', TN, 'FP', FP, 'FN', FN, ...
        'Sensitivity', TP / max(1, sum(y_true==1)), ...
        'Specificity', TN / max(1, sum(y_true==0)), ...
        'Accuracy', mean(y_hat == y_true), ...
        'F1', 2*TP / max(1, 2*TP + FP + FN), ...
        'AUC', local_auc(y_true, scores));
end

function A = local_auc(y_true, scores)
    try
        [~,~,~,A] = perfcurve(y_true, scores, 1);
    catch
        warning('perfcurve not available or failed, set AUC=NaN');
        A = NaN;
    end
end


%%

% clear; clc; close all;
% 
% 
% % load('Balanced_VesselMINST3D_with_history_v2_paper.mat');
% load('Balanced_VesselMINST3D_with_history_v3.mat');


best_iter_lsrtr_manual = 36;  
best_iter_muon_manual = 16;   


fprintf('Extracting performance metrics at iteration %d (LSRTR) and iteration %d (MUON)\n', ...
    best_iter_lsrtr_manual, best_iter_muon_manual);



metrics_at_best_lsrtr = squeeze(all_metrics_history_lsrtr(best_iter_lsrtr_manual, :, :));
metrics_at_best_muon = squeeze(all_metrics_history_muon(best_iter_muon_manual, :, :));


mean_metrics_at_best_lsrtr = mean(metrics_at_best_lsrtr, 1, 'omitnan');
mean_metrics_at_best_muon = mean(metrics_at_best_muon, 1, 'omitnan');


test_error_at_best_lsrtr = mean(all_erry_lsrtr(best_iter_lsrtr_manual, :), 'omitnan');
test_error_at_best_muon = mean(all_erry_muon(best_iter_muon_manual, :), 'omitnan');



time_cumsum_lsrtr = zeros(num_trials, 1);
time_cumsum_muon = zeros(num_trials, 1);

for trial = 1:num_trials
  
    time_cumsum_lsrtr(trial) = sum(all_time_per_iter_lsrtr(1:best_iter_lsrtr_manual, trial), 'omitnan');
    time_cumsum_muon(trial) = sum(all_time_per_iter_muon(1:best_iter_muon_manual, trial), 'omitnan');
end


time_at_best_lsrtr = mean(time_cumsum_lsrtr);
time_at_best_muon = mean(time_cumsum_muon);


fprintf('\nCumulative runtime:\n');
fprintf('  LSRTR (up to iteration %d): %.2f seconds\n', best_iter_lsrtr_manual, time_at_best_lsrtr);
fprintf('  MUON  (up to iteration %d): %.2f seconds\n', best_iter_muon_manual, time_at_best_muon);


figs_path = fullfile('C:\Users\liangx\OneDrive - Iowa State University\Iowa\GLM', 'figure');
figure('Position', [100, 100, 800, 500]);
hold on; grid on; box on;

iters = 1:max_iter;


h1 = plot(iters, mean_erry_lsrtr, '-', 'LineWidth', 2.5, 'Color', [0 0.4470 0.7410]);
h2 = plot(iters, mean_erry_muon,  '--','LineWidth', 2.5, 'Color', [0.8500 0.3250 0.0980]);


xline(best_iter_lsrtr_manual, '--', 'LineWidth', 2, 'Color', [0 0.4470 0.7410], 'Alpha', 0.6);
xline(best_iter_muon_manual, '--', 'LineWidth', 2, 'Color', [0.8500 0.3250 0.0980], 'Alpha', 0.6);


y_max = max([mean_erry_lsrtr; mean_erry_muon], [], 'omitnan');
text(best_iter_lsrtr_manual, y_max*0.95, sprintf('iter %d', best_iter_lsrtr_manual), ...
    'Color', [0 0.4470 0.7410], 'FontSize', 17, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center', 'BackgroundColor', 'white', 'EdgeColor', [0 0.4470 0.7410]);
text(best_iter_muon_manual, y_max*0.85, sprintf('iter %d', best_iter_muon_manual), ...
    'Color', [0.8500 0.3250 0.0980], 'FontSize', 17, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center', 'BackgroundColor', 'white', 'EdgeColor', [0.8500 0.3250 0.0980]);
xlabel('Number of iterations', 'FontSize', 20, 'FontWeight', 'normal');
ylabel('Test error',          'FontSize', 20, 'FontWeight', 'normal');

legend([h1, h2], {'LSRTR', 'LSRTR-M'}, 'FontSize', 17, 'Location', 'best');

set(gca, 'FontSize', 20);

exportgraphics(gcf, fullfile(figs_path, 'Balanced_VesselMINST3D_trial.pdf'), 'ContentType', 'vector');







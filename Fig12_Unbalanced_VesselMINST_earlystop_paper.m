%% VesselMNIST3D: LSRTR vs MUON - 修改版：保存每轮完整数据
clear; clc; close all;
rng(1200);

%% Configuration
load("vesselmnist3d_28.mat");

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
max_iter = 30;
alpha      = 0.7;
alpha_muon = 0.08;
beta       = 0.3;
lam        = 5e-2;


%% Storage for all trials
all_loss_lsrtr = zeros(max_iter, num_trials);
all_loss_muon = zeros(max_iter, num_trials);
all_erry_lsrtr = zeros(max_iter, num_trials);
all_erry_muon = zeros(max_iter, num_trials);


all_metrics_history_lsrtr = nan(max_iter, num_trials, 5); % [Sens, Spec, F1, AUC, Acc]
all_metrics_history_muon = nan(max_iter, num_trials, 5);


all_time_per_iter_lsrtr = zeros(max_iter, num_trials);
all_time_per_iter_muon = zeros(max_iter, num_trials);


all_metrics_lsrtr = zeros(num_trials, 5);
all_metrics_muon = zeros(num_trials, 5);

all_time_lsrtr = zeros(num_trials, 1);
all_time_muon = zeros(num_trials, 1);

actual_iters = zeros(num_trials, 1);

%% Run multiple trials
fprintf('Starting %d trials...\n', num_trials);

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
    
    loss_history = zeros(max_iter, 1);
    loss_history_muon = zeros(max_iter, 1);
    erry_history = zeros(max_iter, 1);
    erry_history_muon = zeros(max_iter, 1);
    
    per_iter_lsrtr = zeros(max_iter, 1);
    per_iter_muon = zeros(max_iter, 1);
    
    B_current = B_0; 
    B_current_muon = B_0; 
    M_current_muon = M_0;
    G_current = G_0;
    G_current_muon = G_0;
    
    %% Training loop
    for iter = 1:max_iter
        if mod(iter, 10) == 0
            fprintf("  iter = %d\n", iter);
        end
        
        %% ========== LSRTR: one full iteration ========== 
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
        
        %% ========== MUON: one full iteration ========== 
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

                vec_B_s_prime_muon   = tilde_B_s_prime_muon * vec_G_current_muon;
                B_s_prime_muon       = reshape(vec_B_s_prime_muon, mm);
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
        G_current_muon      = reshape(vec_G_current_muon, rr);
        
        per_iter_muon(iter) = toc(t1);
        
      
        all_time_per_iter_muon(iter, trial) = per_iter_muon(iter);
        
        % Recompute final vectors
        vec_B_current = tilde_B_current * vec_G_current;
        vec_B_current_muon = tilde_B_current_muon * vec_G_current_muon;
        
        % Compute losses and errors
        eta_train = X * vec_B_current;
        loss_history(iter) = mean(-y_train.*eta_train + log(1 + exp(eta_train)));
        
        eta_train_muon = X * vec_B_current_muon;
        loss_history_muon(iter) = mean(-y_train.*eta_train_muon + log(1 + exp(eta_train_muon)));
        
        p_test = 1./(1 + exp(-(Xt * vec_B_current)));
        y_pred = double(p_test > 0.3);
        erry_history(iter) = mean(abs(y_pred - y_test));
        
        p_test_muon = 1./(1 + exp(-(Xt * vec_B_current_muon)));
        y_pred_muon = double(p_test_muon > 0.3);
        erry_history_muon(iter) = mean(abs(y_pred_muon - y_test));
        
        
        metrics_iter_lsrtr = compute_metrics(y_test, y_pred, p_test);
        all_metrics_history_lsrtr(iter, trial, :) = [metrics_iter_lsrtr.Sensitivity, ...
            metrics_iter_lsrtr.Specificity, metrics_iter_lsrtr.F1, ...
            metrics_iter_lsrtr.AUC, metrics_iter_lsrtr.Accuracy];
        
        metrics_iter_muon = compute_metrics(y_test, y_pred_muon, p_test_muon);
        all_metrics_history_muon(iter, trial, :) = [metrics_iter_muon.Sensitivity, ...
            metrics_iter_muon.Specificity, metrics_iter_muon.F1, ...
            metrics_iter_muon.AUC, metrics_iter_muon.Accuracy];
        % ==========================================
        
        actual_iters(trial) = max_iter;
    end

    % Store results for this trial
    final_iter = actual_iters(trial);
    all_loss_lsrtr(1:final_iter, trial) = loss_history(1:final_iter);
    all_loss_muon(1:final_iter, trial) = loss_history_muon(1:final_iter);
    all_erry_lsrtr(1:final_iter, trial) = erry_history(1:final_iter);
    all_erry_muon(1:final_iter, trial) = erry_history_muon(1:final_iter);
    
    % Store total running time for this trial
    all_time_lsrtr(trial) = sum(per_iter_lsrtr);
    all_time_muon(trial) = sum(per_iter_muon);
    
    % Final evaluation
    p_test_L = 1./(1 + exp(-(Xt * vec_B_current)));
    y_pred_L = double(p_test_L > 0.3);
    
    p_test_M = 1./(1 + exp(-(Xt * vec_B_current_muon)));
    y_pred_M = double(p_test_M > 0.3);
    
    metrics_L = compute_metrics(y_test, y_pred_L, p_test_L);
    metrics_M = compute_metrics(y_test, y_pred_M, p_test_M);
    
    all_metrics_lsrtr(trial, :) = [metrics_L.Sensitivity, metrics_L.Specificity, ...
                                    metrics_L.F1, metrics_L.AUC, metrics_L.Accuracy];
    all_metrics_muon(trial, :) = [metrics_M.Sensitivity, metrics_M.Specificity, ...
                                   metrics_M.F1, metrics_M.AUC, metrics_M.Accuracy];
    
    fprintf('  Trial %d: LSRTR Acc=%.4f (%.2fs), MUON Acc=%.4f (%.2fs)\n', ...
        trial, metrics_L.Accuracy, all_time_lsrtr(trial), ...
        metrics_M.Accuracy, all_time_muon(trial));
end

%% Compute statistics
mean_metrics_lsrtr = mean(all_metrics_lsrtr, 1);
mean_metrics_muon = mean(all_metrics_muon, 1);

mean_erry_lsrtr = mean(all_erry_lsrtr, 2);
mean_erry_muon = mean(all_erry_muon, 2);

mean_time_lsrtr = mean(all_time_lsrtr);
mean_time_muon = mean(all_time_muon);




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





% % load('vesselmnist3d_results_with_history_paper.mat');
% load('vesselmnist3d_results_with_history_v2.mat');


best_iter_muon_manual = 16;    
best_iter_lsrtr_manual = 30;  



fprintf('Extracted performance metrics:\n');
fprintf('  MUON:  iteration %d\n', best_iter_muon_manual);
fprintf('  LSRTR: iteration %d\n\n', best_iter_lsrtr_manual);



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


fprintf('Cumulative runtime:\n');
fprintf('  LSRTR (up to iteration %d): %.2f seconds\n', best_iter_lsrtr_manual, time_at_best_lsrtr);
fprintf('  MUON  (up to iteration %d): %.2f seconds\n\n', best_iter_muon_manual, time_at_best_muon);

% plot
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
% title(sprintf('Average Test Error over %d Trials (Best at iter %d & %d)', ...
%     num_trials, best_iter_lsrtr_manual, best_iter_muon_manual), ...
%     'FontSize', 15, 'FontWeight', 'bold');
legend([h1, h2], {'LSRTR', 'LSRTR-M'}, 'FontSize', 17, 'Location', 'northeast');

set(gca, 'FontSize', 20);
% set(gca, 'YScale', 'log');

% exportgraphics(gcf, fullfile(figs_path, 'Unbalanced_VesselMINST3D_trial.pdf'), 'ContentType', 'vector');



fprintf('Metric               LSRTR (iter %d)  MUON (iter %d)\n', best_iter_lsrtr_manual, best_iter_muon_manual);
fprintf('--------------------------------------------------------\n');
fprintf('%-20s %-17.4f %-17.4f\n', 'Sensitivity', mean_metrics_at_best_lsrtr(1), mean_metrics_at_best_muon(1));
fprintf('%-20s %-17.4f %-17.4f\n', 'Specificity', mean_metrics_at_best_lsrtr(2), mean_metrics_at_best_muon(2));
fprintf('%-20s %-17.4f %-17.4f\n', 'F1 Score', mean_metrics_at_best_lsrtr(3), mean_metrics_at_best_muon(3));
fprintf('%-20s %-17.4f %-17.4f\n', 'AUC', mean_metrics_at_best_lsrtr(4), mean_metrics_at_best_muon(4));
fprintf('%-20s %-17.4f %-17.4f\n', 'Accuracy', mean_metrics_at_best_lsrtr(5), mean_metrics_at_best_muon(5));
fprintf('%-20s %-17.4f %-17.4f\n', 'Test Error', test_error_at_best_lsrtr, test_error_at_best_muon);
fprintf('%-20s %-17.2f %-17.2f\n', 'Running Time (s)', time_at_best_lsrtr, time_at_best_muon);
fprintf('--------------------------------------------------------\n\n');


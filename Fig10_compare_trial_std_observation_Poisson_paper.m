% LSR Tensor Poisson Regression - Multi-trial varying sample size n
% Plots: Final Loss/Error vs Number of training samples (with std shading)
clear; clc; rng(2025); 

%% Parameter Setup
mm = [10 15 20];         % Tensor dimensions
K  = length(mm);         % Tensor order
m  = prod(mm);

rr = [2 2 2];            % LSR structure parameters
r  = prod(rr);
S  = 2;

% Varying number of training samples
n_train_values = [1000, 2000, 3000, 4000, 5000];
num_n_values = length(n_train_values);

% n_test_values  = 0.4*n_train_values;           % # testing samples (fixed)
n_test = 500;


% Algorithm parameters
max_iter     = 20;       % Fixed iterations for each n
alpha        = 0.05;      % RGD stepsize
alpha_muon   = 0.05;     % MUON stepsize
beta         = 0.1;      % MUON momentum
lam          = 1e-3;     % MUON weight decay
perturbation = 0.1;      % init perturb

% Number of trials for each n
num_trials = 50;

% Storage for final results at each n (across trials)
final_loss_rgd = zeros(num_n_values, num_trials);
final_loss_muon = zeros(num_n_values, num_trials);
final_errB_rgd = zeros(num_n_values, num_trials);
final_errB_muon = zeros(num_n_values, num_trials);
final_erry_rgd = zeros(num_n_values, num_trials);
final_erry_muon = zeros(num_n_values, num_trials);

%% Main Loop: Vary n
for n_idx = 1:num_n_values
    n_train = n_train_values(n_idx);
    % n_test  = n_test_values(n_idx);
    fprintf('\n========== n_train = %d ==========\n', n_train);
    
    % Run multiple trials for this n
    for trial = 1:num_trials
        if mod(trial, 10) == 0
            fprintf('Trial %d/%d\n', trial, num_trials);
        end
                
        %% Data Generation 
        % B_true: {B_(k,s)}
        B_true = cell(K, S);
        for s = 1:S
            for k = 1:K
                [Q,R] = qr(randn(mm(k)), 0);
                Q = Q * diag(sign(diag(R) + (diag(R)==0)));
                B_true{k,s} = Q(:, 1:rr(k));
            end
        end
        
        % vec_B_true
        G_true     = randn(rr)/sqrt(r);
        vec_G_true = G_true(:);
        tilde_B    = zeros(m, r);
        for s = 1:S
            Btemp = B_true{K,s};
            for k = (K-1):-1:1
                Btemp = kron(Btemp, B_true{k,s});
            end
            tilde_B = tilde_B + Btemp;
        end
        vec_B_true = tilde_B * vec_G_true;
        
        % Generate training and testing data
        % scale_factor = 1.5 / norm(vec_B_true);  % 调整使 ||vec_B_true|| ≈ 1.5
        % vec_B_true = vec_B_true * scale_factor;
        % G_true = G_true * scale_factor;

        X = randn(n_train, m);
        x_train = X';

        eta = X*vec_B_true;          % n_train x 1 (线性预测器)
        lambda_train = exp(eta);     % 期望（必须为正）
        y_train = poissrnd(lambda_train);  % 泊松分布的计数数据

        Xt = randn(n_test, m);
        x_test = Xt';
        eta_t = Xt*vec_B_true;
        lambda_test = exp(eta_t);
        y_test = poissrnd(lambda_test);
        
        %% Initialization
        G_0 = G_true + perturbation * randn(rr);
        
        B_0 = cell(K, S);
        for s = 1:S
            for k = 1:K
                [Q,R] = qr(B_true{k,s} + perturbation*randn(mm(k), rr(k)), 0);
                Q = Q * diag(sign(diag(R) + (diag(R)==0)));
                B_0{k,s} = Q(:, 1:rr(k));
            end
        end
        
        M_0 = cell(K, S);
        for s = 1:S
            for k = 1:K
                M_0{k,s} = zeros(mm(k), rr(k));
            end
        end
        
        B_current = B_0;
        B_current_muon = B_0;
        M_current_muon = M_0;
        G_current = G_0;
        G_current_muon = G_0;
        
        %% Main Iteration
        for iter = 1:max_iter
            
            %% ========== RGD: one iteration ========== 
            vec_G_current = G_current(:);
            for s_prime = 1:S
                for k_prime = 1:K
                    % tilde_B_current
                    tilde_B_current = zeros(m, r);
                    for s = 1:S
                        Btemp = B_current{K,s};
                        for k = (K-1):-1:1
                            Btemp = kron(Btemp, B_current{k,s});
                        end
                        tilde_B_current = tilde_B_current + Btemp;
                    end
                    vec_B_current = tilde_B_current * vec_G_current;
                    
                    % tilde_B_s_prime
                    tilde_B_s_prime = zeros(m, r);
                    Btemp = B_current{K, s_prime};
                    for k = (K-1):-1:1
                        Btemp = kron(Btemp, B_current{k, s_prime});
                    end
                    tilde_B_s_prime = tilde_B_s_prime + Btemp;
                    vec_B_s_prime = tilde_B_s_prime * vec_G_current;
                    B_s_prime = reshape(vec_B_s_prime, mm);
                    B_s_prime_unfold = mode_k_unfold(B_s_prime, k_prime);
                    G_unfold = mode_k_unfold(G_current, k_prime);
                    
                    % Compute gradient
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

                    val1 = B_current{k_prime,s_prime}*G_unfold*tilde_B_s_prime_neqkp';
                    val2 = B_s_prime_unfold;
                    val3 = norm(val1-val2,'fro')/norm(val2,'fro'); % this is 0, so mode-k unfolding is correct!

                    M_kp_sp = Btemp * G_unfold';
                    Omega_kp_sp = zeros(mm(k_prime)*rr(k_prime), n_train);
                    for i = 1:n_train
                        Xi = reshape(X(i,:).', mm);
                        Xi_unf = mode_k_unfold(Xi, k_prime);
                        Otmp = Xi_unf * M_kp_sp;
                        Omega_kp_sp(:, i) = Otmp(:);
                    end

                    eta_current = X*vec_B_current;
                    lambda_current = exp(eta_current);
                    residual = lambda_current - y_train;


                    vec_grad_B = Omega_kp_sp * residual / n_train;
                    grad_B = reshape(vec_grad_B, mm(k_prime), rr(k_prime));
                    
                    % Update
                    B_temp = B_current{k_prime, s_prime} - alpha * grad_B;
                    [Q, ~] = qr(B_temp, 0);
                    B_current{k_prime, s_prime} = Q;
                end
            end
            
            % RGD: G update
            tilde_B_current = zeros(m, r);
            for s = 1:S
                Btemp = B_current{K,s};
                for k = (K-1):-1:1
                    Btemp = kron(Btemp, B_current{k,s});
                end
                tilde_B_current = tilde_B_current + Btemp;
            end
            vec_B_current = tilde_B_current*vec_G_current;
            eta_current = X*vec_B_current;
            lambda_current = exp(eta_current);
            residual = lambda_current - y_train;

            XB = X * tilde_B_current;
            vec_grad_G = (XB)' * residual / n_train;
            vec_G_current = vec_G_current - alpha * vec_grad_G;
            G_current = reshape(vec_G_current, rr);
            
            %% ========== MUON: one iteration ========== 
            vec_G_current_muon = G_current_muon(:);
            for s_prime = 1:S
                for k_prime = 1:K
                    % tilde_B_current_muon
                    tilde_B_current_muon = zeros(m, r);
                    for s = 1:S
                        Btemp = B_current_muon{K,s};
                        for k = (K-1):-1:1
                            Btemp = kron(Btemp, B_current_muon{k,s});
                        end
                        tilde_B_current_muon = tilde_B_current_muon + Btemp;
                    end
                    vec_B_current_muon = tilde_B_current_muon * vec_G_current_muon;
                    
                    % tilde_B_s_prime_muon
                    tilde_B_s_prime_muon = zeros(m, r);
                    Btemp = B_current_muon{K, s_prime};
                    for k = (K-1):-1:1
                        Btemp = kron(Btemp, B_current_muon{k, s_prime});
                    end
                    tilde_B_s_prime_muon = tilde_B_s_prime_muon + Btemp;
                    vec_B_s_prime_muon = tilde_B_s_prime_muon * vec_G_current_muon;
                    B_s_prime_muon = reshape(vec_B_s_prime_muon, mm);
                    B_s_prime_muon_unfold = mode_k_unfold(B_s_prime_muon, k_prime);
                    G_muon_unfold = mode_k_unfold(G_current_muon, k_prime);
                    
                    % Compute gradient
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

                    val1 = B_current_muon{k_prime,s_prime}*G_muon_unfold*tilde_B_s_prime_neqkp_muon';
                    val2 = B_s_prime_muon_unfold;
                    val3 = norm(val1-val2,'fro')/norm(val2,'fro'); % this is 0, so mode-k unfolding is correct!

                    
                    M_kp_sp_muon = Btemp * G_muon_unfold';
                    Omega_kp_sp_muon = zeros(mm(k_prime)*rr(k_prime), n_train);
                    for i = 1:n_train
                        Xi = reshape(X(i,:).', mm);
                        Xi_unf = mode_k_unfold(Xi, k_prime);
                        Otmp = Xi_unf * M_kp_sp_muon;
                        Omega_kp_sp_muon(:, i) = Otmp(:);
                    end
                    
                    eta_muon = X*vec_B_current_muon;
                    lambda_muon = exp(eta_muon);
                    residual_muon = lambda_muon - y_train;

                    vec_grad_B_muon = Omega_kp_sp_muon * residual_muon / n_train;
                    grad_B_muon = reshape(vec_grad_B_muon, mm(k_prime), rr(k_prime));
                    M_current_muon{k_prime, s_prime} = beta*M_current_muon{k_prime, s_prime} + grad_B_muon;
                    
                    % MUON update
                    eps_reg = 1e-4;
                    Mcm = M_current_muon{k_prime, s_prime};
                    Acm = Mcm' * Mcm + eps_reg * eye(size(Mcm,2));
                    [Ucm, Scm] = eig(Acm);
                    A_invhalf = Ucm * diag(1./sqrt(diag(Scm))) * Ucm';
                    Qcm = Mcm * A_invhalf;
                    B_current_muon{k_prime, s_prime} = B_current_muon{k_prime, s_prime} - alpha_muon*(Qcm + lam*B_current_muon{k_prime, s_prime});
         
                end
            end
            
            % MUON: G update
            tilde_B_current_muon = zeros(m, r);
            for s = 1:S
                Btemp = B_current_muon{K,s};
                for k = (K-1):-1:1
                    Btemp = kron(Btemp, B_current_muon{k,s});
                end
                tilde_B_current_muon = tilde_B_current_muon + Btemp;
            end
            vec_B_current_muon = tilde_B_current_muon*vec_G_current_muon;

            eta_muon = X*vec_B_current_muon;
            lambda_muon = exp(eta_muon);
            residual_muon = lambda_muon - y_train;

            XB = X * tilde_B_current_muon;
            vec_grad_G_muon = (XB)' * residual_muon / n_train;
            vec_G_current_muon = vec_G_current_muon - alpha * vec_grad_G_muon;
            G_current_muon = reshape(vec_G_current_muon, rr);
        end
        
        %% Store Final Results (after max_iter iterations)
        vec_B_current = tilde_B_current * vec_G_current;
        vec_B_current_muon = tilde_B_current_muon * vec_G_current_muon;


        eta_test = Xt*vec_B_current;
        lambda_test_pred = exp(eta_test);
        y_pred =  poissrnd(lambda_test_pred); 

        eta_test_muon = Xt*vec_B_current_muon;
        lambda_test_pred_muon = exp(eta_test_muon);
        y_pred_muon = poissrnd(lambda_test_pred_muon);

        eta_train = X*vec_B_current;
        final_loss_rgd(n_idx, trial) = mean(-y_train.*eta_train + exp(eta_train));
        final_errB_rgd(n_idx, trial) = norm(vec_B_current - vec_B_true)^2/norm(vec_B_true)^2;
        final_erry_rgd(n_idx, trial) = norm(log(y_pred + 1) - log(y_test + 1))^2 / norm(log(y_test + 1))^2;
        
        eta_train_muon = X*vec_B_current_muon;
        final_loss_muon(n_idx, trial) = mean(-y_train.*eta_train_muon + exp(eta_train_muon));
        final_errB_muon(n_idx, trial) = norm(vec_B_current_muon - vec_B_true)^2/norm(vec_B_true)^2;
        final_erry_muon(n_idx, trial) = norm(log(y_pred_muon + 1) - log(y_test + 1))^2 / norm(log(y_test + 1))^2;
    end
end

% Compute Statistics
mean_loss_rgd = mean(final_loss_rgd, 2);
std_loss_rgd = std(final_loss_rgd, 0, 2);
mean_loss_muon = mean(final_loss_muon, 2);
std_loss_muon = std(final_loss_muon, 0, 2);

mean_errB_rgd = mean(final_errB_rgd, 2);
std_errB_rgd = std(final_errB_rgd, 0, 2);
mean_errB_muon = mean(final_errB_muon, 2);
std_errB_muon = std(final_errB_muon, 0, 2);

mean_erry_rgd = mean(final_erry_rgd, 2);
std_erry_rgd = std(final_erry_rgd, 0, 2);
mean_erry_muon = mean(final_erry_muon, 2);
std_erry_muon = std(final_erry_muon, 0, 2);



%% 
% clc;clear
% figs_path = fullfile('C:\Users\liangx\OneDrive - Iowa State University\Iowa\GLM', 'figure');
% % Plotting Section - Modified to handle NaN values
% load("compare_trial_std_observation_Poisson_120815_paper.mat")

% Filter out NaN trials and compute statistics
mean_loss_rgd = zeros(num_n_values, 1);
std_loss_rgd = zeros(num_n_values, 1);
mean_loss_muon = zeros(num_n_values, 1);
std_loss_muon = zeros(num_n_values, 1);

mean_errB_rgd = zeros(num_n_values, 1);
std_errB_rgd = zeros(num_n_values, 1);
mean_errB_muon = zeros(num_n_values, 1);
std_errB_muon = zeros(num_n_values, 1);

mean_erry_rgd = zeros(num_n_values, 1);
std_erry_rgd = zeros(num_n_values, 1);
mean_erry_muon = zeros(num_n_values, 1);
std_erry_muon = zeros(num_n_values, 1);

% Convergence rate tracking
conv_rate_rgd = zeros(num_n_values, 1);
conv_rate_muon = zeros(num_n_values, 1);
conv_count_rgd = zeros(num_n_values, 1);
conv_count_muon = zeros(num_n_values, 1);

for n_idx = 1:num_n_values
    % RGD convergence check (not NaN only)
    valid_rgd = ~isnan(final_loss_rgd(n_idx, :)) & ...
                ~isnan(final_errB_rgd(n_idx, :)) & ...
                ~isnan(final_erry_rgd(n_idx, :));
    conv_count_rgd(n_idx) = sum(valid_rgd);
    conv_rate_rgd(n_idx) = conv_count_rgd(n_idx) / num_trials * 100;
    
    if conv_count_rgd(n_idx) > 0
        mean_loss_rgd(n_idx) = mean(final_loss_rgd(n_idx, valid_rgd));
        std_loss_rgd(n_idx) = std(final_loss_rgd(n_idx, valid_rgd));
        mean_errB_rgd(n_idx) = mean(final_errB_rgd(n_idx, valid_rgd));
        std_errB_rgd(n_idx) = std(final_errB_rgd(n_idx, valid_rgd));
        mean_erry_rgd(n_idx) = mean(final_erry_rgd(n_idx, valid_rgd));
        std_erry_rgd(n_idx) = std(final_erry_rgd(n_idx, valid_rgd));
    else
        mean_loss_rgd(n_idx) = NaN;
        std_loss_rgd(n_idx) = NaN;
        mean_errB_rgd(n_idx) = NaN;
        std_errB_rgd(n_idx) = NaN;
        mean_erry_rgd(n_idx) = NaN;
        std_erry_rgd(n_idx) = NaN;
    end
    
    % MUON convergence check (not NaN and final_errB_muon < 0.01)
    valid_muon = ~isnan(final_loss_muon(n_idx, :)) & ...
                 ~isnan(final_errB_muon(n_idx, :)) & ...
                 ~isnan(final_erry_muon(n_idx, :)) & ...
                 (final_errB_muon(n_idx, :) < 0.1);
    conv_count_muon(n_idx) = sum(valid_muon);
    conv_rate_muon(n_idx) = conv_count_muon(n_idx) / num_trials * 100;
    
    if conv_count_muon(n_idx) > 0
        mean_loss_muon(n_idx) = mean(final_loss_muon(n_idx, valid_muon));
        std_loss_muon(n_idx) = std(final_loss_muon(n_idx, valid_muon));
        mean_errB_muon(n_idx) = mean(final_errB_muon(n_idx, valid_muon));
        std_errB_muon(n_idx) = std(final_errB_muon(n_idx, valid_muon));
        mean_erry_muon(n_idx) = mean(final_erry_muon(n_idx, valid_muon));
        std_erry_muon(n_idx) = std(final_erry_muon(n_idx, valid_muon));
    else
        mean_loss_muon(n_idx) = NaN;
        std_loss_muon(n_idx) = NaN;
        mean_errB_muon(n_idx) = NaN;
        std_errB_muon(n_idx) = NaN;
        mean_erry_muon(n_idx) = NaN;
        std_erry_muon(n_idx) = NaN;
    end
    
    fprintf('n=%d: LSRTR converged %d/%d (%.1f%%), Muon converged %d/%d (%.1f%%)\n', ...
            n_train_values(n_idx), conv_count_rgd(n_idx), num_trials, conv_rate_rgd(n_idx), ...
            conv_count_muon(n_idx), num_trials, conv_rate_muon(n_idx));
end

% Plotting
fs = 20;

% % Figure 1: Convergence Rate Comparison
figure(4); clf;
hold on;
plot(n_train_values, conv_rate_rgd, 'b-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'LSRTR');
plot(n_train_values, conv_rate_muon, 'r-s', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'LSRTR-M');
xlabel('Number of observations (\times 10^4)', 'Interpreter','tex', 'FontSize', fs);
ylabel('Convergence success rate', 'FontSize', fs);
legend('show', 'Location', 'southeast', 'FontSize', fs-3);
set(gcf, 'Color', 'w');
set(gca, 'FontSize', fs);
ylim([0, 105]);
grid on;

ax = gca;


ax.XTick = [5000 10000 15000 20000 25000];


ax.XAxis.Exponent = 0;     
ax.XTickLabel = compose('%.1f', ax.XTick/1e4);
ax.XTickLabel = strrep(ax.XTickLabel,'.0',''); 
exportgraphics(gcf, fullfile(figs_path, 'convergence_rate_vs_n_Poisson.pdf'), 'ContentType', 'vector');
% 
% 
% % Figure 1.5: Convergence Statistics Table
% figure(7); clf;
% % Create table data
% table_data = cell(num_n_values+1, 3);
% table_data{1,1} = 'n';
% table_data{1,2} = 'LSRTR';
% table_data{1,3} = 'Muon';
% 
% for n_idx = 1:num_n_values
%     table_data{n_idx+1, 1} = sprintf('%d', n_train_values(n_idx));
%     table_data{n_idx+1, 2} = sprintf('%d/%d (%.1f%%)', ...
%         conv_count_rgd(n_idx), num_trials, conv_rate_rgd(n_idx));
%     table_data{n_idx+1, 3} = sprintf('%d/%d (%.1f%%)', ...
%         conv_count_muon(n_idx), num_trials, conv_rate_muon(n_idx));
% end
% 
% % Create figure with table
% axis off;
% t = uitable('Parent', gcf, 'Data', table_data(2:end,:), ...
%     'ColumnName', table_data(1,:), ...
%     'Units', 'normalized', ...
%     'Position', [0.1 0.2 0.8 0.7], ...
%     'FontSize', 14, ...
%     'RowName', [], ...
%     'ColumnWidth', {100, 180, 180});
% 
% % Add title
% annotation('textbox', [0.1, 0.9, 0.8, 0.08], ...
%     'String', 'Convergence Statistics (Muon: errB < 0.01)', ...
%     'FontSize', 16, 'FontWeight', 'bold', ...
%     'HorizontalAlignment', 'center', ...
%     'EdgeColor', 'none');
% 
% set(gcf, 'Color', 'w', 'Position', [100, 100, 600, 400]);
% % exportgraphics(gcf, fullfile(figs_path, 'convergence_table_Poisson.pdf'), 'ContentType', 'vector');


% Figure 2: Parameter Error vs Sample Size
figure(5); clf;
hold on;
% LSRTR
valid_idx_rgd = ~isnan(mean_errB_rgd);
lower_errB_rgd = max(mean_errB_rgd(valid_idx_rgd) - std_errB_rgd(valid_idx_rgd), 1e-10);
upper_errB_rgd = mean_errB_rgd(valid_idx_rgd) + std_errB_rgd(valid_idx_rgd);
fill([n_train_values(valid_idx_rgd), fliplr(n_train_values(valid_idx_rgd))], ...
     [upper_errB_rgd; flipud(lower_errB_rgd)]', ...
     'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(n_train_values(valid_idx_rgd), mean_errB_rgd(valid_idx_rgd), 'b-o', ...
     'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'LSRTR');
% Muon
valid_idx_muon = ~isnan(mean_errB_muon);
lower_errB_muon = max(mean_errB_muon(valid_idx_muon) - std_errB_muon(valid_idx_muon), 1e-10);
upper_errB_muon = mean_errB_muon(valid_idx_muon) + std_errB_muon(valid_idx_muon);
fill([n_train_values(valid_idx_muon), fliplr(n_train_values(valid_idx_muon))], ...
     [upper_errB_muon; flipud(lower_errB_muon)]', ...
     'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(n_train_values(valid_idx_muon), mean_errB_muon(valid_idx_muon), 'r-s', ...
     'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'LSRTR-M');
xlabel('Number of observations (\times 10^4)', 'Interpreter','tex', 'FontSize', fs);
% ylabel('$\|\widehat{\underline{\mathbf{B}}}-\underline{\mathbf{B}}\|_F^2/\|\underline{\mathbf{B}}\|_F^2$', ...
%        'Interpreter', 'latex', 'FontSize', fs);
ylabel('$\|\widehat{\underline{\mathcal{B}}}-\underline{\mathcal{B}}\|_F^2/\|\underline{\mathcal{B}}\|_F^2$', ...
      'Interpreter', 'latex', 'FontSize', fs);
legend('show', 'Location', 'northeast', 'FontSize', fs-3);
set(gcf, 'Color', 'w');
set(gca, 'FontSize', fs);
set(gca, 'YScale', 'log');
axis tight;
grid on;
% ax = gca;                     % 当前坐标轴
% ax.XAxis.Exponent = 0;        % 不用 10^4 这种指数
% ax.XTick = [10000 20000];         
% ax.XTickLabel = compose('%d', ax.XTick);
ax = gca;

% 显示全部横坐标刻度：5000,10000,15000,20000,25000
ax.XTick = [5000 10000 15000 20000 25000];

% 让 MATLAB 用科学计数法，把 ×10^4 放在坐标轴旁
ax.XAxis.Exponent = 0;     % 关键：显示成 0.5,1,1.5,2,2.5，并在轴边显示 ×10^4
ax.XTickLabel = compose('%.1f', ax.XTick/1e4);
ax.XTickLabel = strrep(ax.XTickLabel,'.0','');  % 可选：把 1.0 变成 1
exportgraphics(gcf, fullfile(figs_path, 'estimation_vs_n_Poisson.pdf'), 'ContentType', 'vector');


% Figure 3: Prediction Error vs Sample Size
figure(6); clf;
hold on;
% LSRTR
valid_idx_rgd = ~isnan(mean_erry_rgd);
lower_erry_rgd = max(mean_erry_rgd(valid_idx_rgd) - std_erry_rgd(valid_idx_rgd), 0);
upper_erry_rgd = mean_erry_rgd(valid_idx_rgd) + std_erry_rgd(valid_idx_rgd);
fill([n_train_values(valid_idx_rgd), fliplr(n_train_values(valid_idx_rgd))], ...
     [upper_erry_rgd; flipud(lower_erry_rgd)]', ...
     'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(n_train_values(valid_idx_rgd), mean_erry_rgd(valid_idx_rgd), 'b-o', ...
     'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'LSRTR');
% Muon
valid_idx_muon = ~isnan(mean_erry_muon);
lower_erry_muon = max(mean_erry_muon(valid_idx_muon) - std_erry_muon(valid_idx_muon), 0);
upper_erry_muon = mean_erry_muon(valid_idx_muon) + std_erry_muon(valid_idx_muon);
fill([n_train_values(valid_idx_muon), fliplr(n_train_values(valid_idx_muon))], ...
     [upper_erry_muon; flipud(lower_erry_muon)]', ...
     'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(n_train_values(valid_idx_muon), mean_erry_muon(valid_idx_muon), 'r-s', ...
     'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'LSRTR-M');
xlabel('Number of observations (\times 10^4)', 'Interpreter','tex', 'FontSize', fs);
ylabel(['$\frac{\|\log(\hat{\mathbf{y}}+1)-\log(\mathbf{y}+1)\|_{F}^2}' ...
        '{\|\log(\mathbf{y}+1)\|_{F}^2}$'], ...
       'Interpreter','latex','FontSize',fs);
legend('show', 'Location', 'northeast', 'FontSize', fs-3);
set(gcf, 'Color', 'w');
set(gca, 'FontSize', fs);
set(gca, 'YScale', 'log');
axis tight;
grid on;

% ax = gca;                     % 当前坐标轴
% ax.XAxis.Exponent = 0;        % 不用 10^4 这种指数
% ax.XTick = [10000 20000];         
% ax.XTickLabel = compose('%d', ax.XTick);
ax = gca;

% 显示全部横坐标刻度：5000,10000,15000,20000,25000
ax.XTick = [5000 10000 15000 20000 25000];

% 让 MATLAB 用科学计数法，把 ×10^4 放在坐标轴旁
ax.XAxis.Exponent = 0;     % 关键：显示成 0.5,1,1.5,2,2.5，并在轴边显示 ×10^4
ax.XTickLabel = compose('%.1f', ax.XTick/1e4);
ax.XTickLabel = strrep(ax.XTickLabel,'.0','');  % 可选：把 1.0 变成 1
exportgraphics(gcf, fullfile(figs_path, 'prediction_vs_n_Poisson.pdf'), 'ContentType', 'vector');

%% 
% figs_path = fullfile('C:\Users\liangx\OneDrive - Iowa State University\Iowa\GLM', 'figure');
% 
% % exportgraphics(gcf, fullfile(figs_path, 'convergence_rate_vs_n_Poisson.pdf'), 'ContentType', 'vector');
% exportgraphics(gcf, fullfile(figs_path, 'estimation_vs_n_Poisson.pdf'), 'ContentType', 'vector');
% exportgraphics(gcf, fullfile(figs_path, 'prediction_vs_n_Poisson.pdf'), 'ContentType', 'vector');

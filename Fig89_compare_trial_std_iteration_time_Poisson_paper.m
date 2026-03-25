% LSR Tensor Linear Regression - Multi-trial with Time and Iteration Analysis
% Plots: Loss/Error vs Iterations (with std) AND Loss/Error vs Time (mean only)
clear; clc;rng(2025);

%% Parameter Setup
mm = [10 15 20];         % Tensor dimensions
K  = length(mm);         % Tensor order
m  = prod(mm);

rr = [2 2 2];            % LSR structure parameters
r  = prod(rr);
S  = 2;

n_train = 5000;           % # training samples
n_test  = 1000;           % # testing samples


% Algorithm parameters
max_iter     = 20;
alpha        = 0.05;      % RGD stepsize
alpha_muon   = 0.05;     % MUON stepsize
beta         = 0.1;      % MUON momentum
lam          = 1e-3;     % MUON weight decay
perturbation = 0.1;      % init perturb
tol          = 1e-35;    % stop criteria (disabled for fixed iterations)

% Number of trials
num_trials = 50;

% Storage for all trials
all_loss_history = zeros(max_iter, num_trials);
all_loss_history_muon = zeros(max_iter, num_trials);
all_errB_history = zeros(max_iter, num_trials);
all_errB_history_muon = zeros(max_iter, num_trials);
all_erry_history = zeros(max_iter, num_trials);
all_erry_history_muon = zeros(max_iter, num_trials);

% Time storage
all_time_rgd = zeros(max_iter, num_trials);
all_time_muon = zeros(max_iter, num_trials);

%% Run Multiple Trials
for trial = 1:num_trials
    fprintf('========== Trial %d/%d ==========\n', trial, num_trials);
    
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
    
    % Initialize histories
    loss_history = zeros(max_iter, 1);
    errB_history = zeros(max_iter, 1);
    erry_history = zeros(max_iter, 1);
    loss_history_muon = zeros(max_iter, 1);
    errB_history_muon = zeros(max_iter, 1);
    erry_history_muon = zeros(max_iter, 1);
    
    per_iter_rgd  = zeros(max_iter, 1);
    per_iter_muon = zeros(max_iter, 1);
    
    B_current = B_0;
    B_current_muon = B_0;
    M_current_muon = M_0;
    G_current = G_0;
    G_current_muon = G_0;
    
    %% Main Iteration
    for iter = 1:max_iter
        if mod(iter, 10) == 0
            fprintf("  iter = %d\n", iter);
        end
        
        %% ========== RGD: one full iteration ========== 
        t0 = tic;
        
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
        
        per_iter_rgd(iter) = toc(t0);
        
        %% ========== MUON: one full iteration ========== 
        t1 = tic;
        
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
        
        per_iter_muon(iter) = toc(t1);
        
        %% Compute Metrics
        vec_B_current = tilde_B_current * vec_G_current;
        vec_B_current_muon = tilde_B_current_muon * vec_G_current_muon;

        %LSRTR
        eta_train = X*vec_B_current;
        loss_history(iter) = mean(-y_train.*eta_train + exp(eta_train));
        errB_history(iter) = norm(vec_B_current - vec_B_true)^2/norm(vec_B_true)^2;
        eta_test = Xt*vec_B_current;
        lambda_test_pred = exp(eta_test);
        y_pred = lambda_test_pred;  % 使用期望值
        erry_history(iter) = norm(log(y_pred + 1) - log(y_test + 1))^2 / norm(log(y_test + 1))^2;
        
        %MUON
        eta_train_muon = X*vec_B_current_muon;
        loss_history_muon(iter) = mean(-y_train.*eta_train_muon + exp(eta_train_muon));

        % MUON: 参数误差
        errB_history_muon(iter) = norm(vec_B_current_muon - vec_B_true)^2/norm(vec_B_true)^2;

        % MUON: 测试集 normalised squared logarithmic error
        eta_test_muon = Xt*vec_B_current_muon;
        lambda_test_pred_muon = exp(eta_test_muon);
        y_pred_muon = lambda_test_pred_muon;
        erry_history_muon(iter) = norm(log(y_pred_muon + 1) - log(y_test + 1))^2 / norm(log(y_test + 1))^2;
        
       
       
    end
    
    % Store results for this trial
    all_loss_history(:, trial) = loss_history;
    all_loss_history_muon(:, trial) = loss_history_muon;
    all_errB_history(:, trial) = errB_history;
    all_errB_history_muon(:, trial) = errB_history_muon;
    all_erry_history(:, trial) = erry_history;
    all_erry_history_muon(:, trial) = erry_history_muon;
    all_time_rgd(:, trial) = cumsum(per_iter_rgd);
    all_time_muon(:, trial) = cumsum(per_iter_muon);
end

%% Compute Statistics
% For iteration plots (with std)
mean_loss = mean(all_loss_history, 2);
std_loss = std(all_loss_history, 0, 2);
mean_loss_muon = mean(all_loss_history_muon, 2);
std_loss_muon = std(all_loss_history_muon, 0, 2);

mean_errB = mean(all_errB_history, 2);
std_errB = std(all_errB_history, 0, 2);
mean_errB_muon = mean(all_errB_history_muon, 2);
std_errB_muon = std(all_errB_history_muon, 0, 2);

mean_erry = mean(all_erry_history, 2);
std_erry = std(all_erry_history, 0, 2);
mean_erry_muon = mean(all_erry_history_muon, 2);
std_erry_muon = std(all_erry_history_muon, 0, 2);

% For time plots (mean only, no std)
mean_time_rgd = mean(all_time_rgd, 2);
mean_time_muon = mean(all_time_muon, 2);

iters = 1:max_iter;

%% Plotting
% clc;clear;close all;
% figs_path = fullfile('C:\Users\liangx\OneDrive - Iowa State University\Iowa\GLM', 'figure');
% % figs_path = fullfile('C:\Users\l.xiao\OneDrive - Iowa State University\Iowa\GLM', 'figure');
% load ("Poisson_trial_iteration_time_120223_paper.mat")

% if ~exist(figs_path,'dir'), mkdir(figs_path); end

fs = 20;

% ========== ITERATION PLOTS (with std shading) ==========

% Figure 1: Loss vs Iterations
figure(1); clf;
hold on;
% LSRTR
lower_loss = max(mean_loss - std_loss, 0);
upper_loss = mean_loss + std_loss;
fill([iters, fliplr(iters)], [upper_loss; flipud(mean_loss - std_loss)]', ...
     'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(iters, mean_loss, 'b-', 'LineWidth', 2, 'DisplayName', 'LSRTR');
% Muon
% lower_loss_muon = max(mean_loss_muon - std_loss_muon, 0);
lower_loss_muon = mean_loss_muon - std_loss_muon;
upper_loss_muon = mean_loss_muon + std_loss_muon;
fill([iters, fliplr(iters)], [upper_loss_muon; flipud(lower_loss_muon)]', ...
     'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(iters, mean_loss_muon, 'r-', 'LineWidth', 2, 'DisplayName', 'LSRTR-M');
xlabel('Number of iterations', 'FontSize', fs);
ylabel('Loss', 'FontSize', fs);
legend('show', 'Location', 'northeast', 'FontSize', fs-3);
set(gcf, 'Color', 'w');
set(gca, 'FontSize', fs);
% set(gca, 'YScale', 'log');
ylim([1e-10, inf]);
axis tight;
grid on;
%title(sprintf('Loss vs Iterations (%d trials)', num_trials), 'FontSize', fs-2);
% exportgraphics(gcf, fullfile(figs_path, 'loss_vs_iter_Poisson_1.pdf'), 'ContentType', 'vector');


% Figure 2: Parameter Error vs Iterations
figure(2); clf;
hold on;
lower_errB = max(mean_errB - std_errB, 1e-10);
% lower_errB = mean_errB - std_errB;
upper_errB = mean_errB + std_errB;
fill([iters, fliplr(iters)], [upper_errB; flipud(lower_errB)]', ...
     'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(iters, mean_errB, 'b-', 'LineWidth', 2, 'DisplayName', 'LSRTR');

lower_errB_muon = max(mean_errB_muon - std_errB_muon, 1e-10);
upper_errB_muon = mean_errB_muon + std_errB_muon;
fill([iters, fliplr(iters)], [upper_errB_muon; flipud(lower_errB_muon)]', ...
     'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(iters, mean_errB_muon, 'r-', 'LineWidth', 2, 'DisplayName', 'LSRTR-M');
xlabel('Number of iterations', 'FontSize', fs);
% ylabel('$\|\widehat{\underline{\mathbf{B}}}-\underline{\mathbf{B}}\|_F^2/\|\underline{\mathbf{B}}\|_F^2$', 'Interpreter', 'latex', 'FontSize', fs);
ylabel('$\|\widehat{\underline{\mathcal{B}}}-\underline{\mathcal{B}}\|_F^2/\|\underline{\mathcal{B}}\|_F^2$', ...
      'Interpreter', 'latex', 'FontSize', fs);
legend('show', 'Location', 'northeast', 'FontSize', fs-3);
set(gcf, 'Color', 'w');
set(gca, 'FontSize', fs);
set(gca, 'YScale', 'log');
ylim([1e-10, inf]);
axis tight;
grid on;
%title(sprintf('Parameter Error vs Iterations (%d trials)', num_trials), 'FontSize', fs-2);
% exportgraphics(gcf, fullfile(figs_path, 'estimation_vs_iter_Poisson_1.pdf'), 'ContentType', 'vector');


% Figure 3: Prediction Error vs Iterations
figure(3); clf;
hold on;
lower_erry = max(mean_erry - std_erry, 0);
upper_erry = mean_erry + std_erry;
fill([iters, fliplr(iters)], [upper_erry; flipud(lower_erry)]', ...
     'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(iters, mean_erry, 'b-', 'LineWidth', 2, 'DisplayName', 'LSRTR');
lower_erry_muon = max(mean_erry_muon - std_erry_muon, 0);
upper_erry_muon = mean_erry_muon + std_erry_muon;
fill([iters, fliplr(iters)], [upper_erry_muon; flipud(lower_erry_muon)]', ...
     'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(iters, mean_erry_muon, 'r-', 'LineWidth', 2, 'DisplayName', 'LSRTR-M');
xlabel('Number of iterations', 'FontSize', fs);
% ylabel('$\|\log(\widehat{\mathbf{y}}+1)-\log(\mathbf{y}+1)\|_2^2/\|\log(\mathbf{y}+1)\|_2^2$','Interpreter', 'latex','FontSize', fs);
ylabel(['$\frac{\|\log(\hat{\mathbf{y}}+1)-\log(\mathbf{y}+1)\|_{F}^2}' ...
        '{\|\log(\mathbf{y}+1)\|_{F}^2}$'], ...
       'Interpreter','latex','FontSize',fs);
legend('show', 'Location', 'northeast', 'FontSize', fs-3);
set(gcf, 'Color', 'w');
set(gca, 'FontSize', fs);
% set(gca, 'YScale', 'log');
ylim([1e-10, inf]);
axis tight;
grid on;
%title(sprintf('Prediction Error vs Iterations (%d trials)', num_trials), 'FontSize', fs-2);
% exportgraphics(gcf, fullfile(figs_path, 'prediction_vs_iter_Poisson_1.pdf'), 'ContentType', 'vector');

% ========== TIME PLOTS (mean only, no std) ==========

% Markers for visibility
skip1 = max(1, floor(max_iter/20));
idx_markers = 1:skip1:max_iter;

% Figure 4: Loss vs Time
figure(4); clf;
hold on;
plot(mean_time_rgd, mean_loss, 'b-+', 'LineWidth', 2, ...
     'MarkerIndices', idx_markers, 'MarkerSize', 6, 'DisplayName', 'LSRTR');
plot(mean_time_muon, mean_loss_muon, 'r-*', 'LineWidth', 2, ...
     'MarkerIndices', idx_markers, 'MarkerSize', 6, 'DisplayName', 'LSRTR-M');
xlabel('Running time (s)', 'FontSize', fs);
ylabel('Loss', 'FontSize', fs);
legend('show', 'Location', 'northeast', 'FontSize', fs-3);
set(gcf, 'Color', 'w');
set(gca, 'FontSize', fs);
% set(gca, 'YScale', 'log');
axis tight;
grid on;
%title(sprintf('Loss vs Time (%d trials, mean)', num_trials), 'FontSize', fs-2);
% exportgraphics(gcf, fullfile(figs_path, 'loss_vs_time_Poisson_1.pdf'), 'ContentType', 'vector');

% Figure 5: Parameter Error vs Time
figure(5); clf;
hold on;
plot(mean_time_rgd, mean_errB, 'b-+', 'LineWidth', 2, ...
     'MarkerIndices', idx_markers, 'MarkerSize', 6, 'DisplayName', 'LSRTR');
plot(mean_time_muon, mean_errB_muon, 'r-*', 'LineWidth', 2, ...
     'MarkerIndices', idx_markers, 'MarkerSize', 6, 'DisplayName', 'LSRTR-M');
xlabel('Running time (s)', 'FontSize', fs);
% ylabel('$\|\widehat{\underline{\mathbf{B}}}-\underline{\mathbf{B}}\|_F^2/\|\underline{\mathbf{B}}\|_F^2$', 'Interpreter', 'latex', 'FontSize', fs);
ylabel('$\|\widehat{\underline{\mathcal{B}}}-\underline{\mathcal{B}}\|_F^2/\|\underline{\mathcal{B}}\|_F^2$', ...
      'Interpreter', 'latex', 'FontSize', fs);
legend('show', 'Location', 'northeast', 'FontSize', fs-3);
set(gcf, 'Color', 'w');
set(gca, 'FontSize', fs);
set(gca, 'YScale', 'log');
axis tight;
grid on;
%title(sprintf('Parameter Error vs Time (%d trials, mean)', num_trials), 'FontSize', fs-2);
% exportgraphics(gcf, fullfile(figs_path, 'estimation_vs_time_Poisson_1.pdf'), 'ContentType', 'vector');

% Figure 6: Prediction Error vs Time
figure(6); clf;
hold on;
plot(mean_time_rgd, mean_erry, 'b-+', 'LineWidth', 2, ...
     'MarkerIndices', idx_markers, 'MarkerSize', 6, 'DisplayName', 'LSRTR');
plot(mean_time_muon, mean_erry_muon, 'r-*', 'LineWidth', 2, ...
     'MarkerIndices', idx_markers, 'MarkerSize', 6, 'DisplayName', 'LSRTR-M');
xlabel('Running time (s)', 'FontSize', fs);
% ylabel('$\|\log(\widehat{\mathbf{y}}+1)-\log(\mathbf{y}+1)\|_2^2/\|\log(\mathbf{y}+1)\|_2^2$','Interpreter', 'latex','FontSize', fs);
ylabel(['$\frac{\|\log(\hat{\mathbf{y}}+1)-\log(\mathbf{y}+1)\|_{F}^2}' ...
        '{\|\log(\mathbf{y}+1)\|_{F}^2}$'], ...
       'Interpreter','latex','FontSize',fs);
legend('show', 'Location', 'northeast', 'FontSize', fs-3);
set(gcf, 'Color', 'w');
set(gca, 'FontSize', fs);
% set(gca, 'YScale', 'log');
axis tight;
grid on;
%title(sprintf('Prediction Error vs Time (%d trials, mean)', num_trials), 'FontSize', fs-2);
% exportgraphics(gcf, fullfile(figs_path, 'prediction_vs_time_Poisson_1.pdf'), 'ContentType', 'vector');
% 
% 
% %% Summary
% fprintf('\n========== Summary ==========\n');
% fprintf('Completed %d trials, %d iterations each\n', num_trials, max_iter);
% fprintf('\nFinal Results (mean ± std):\n');
% fprintf('LSRTR - Loss: %.6e ± %.6e\n', mean_loss(end), std_loss(end));
% fprintf('Muon  - Loss: %.6e ± %.6e\n', mean_loss_muon(end), std_loss_muon(end));
% fprintf('LSRTR - Param Error: %.6e ± %.6e\n', mean_errB(end), std_errB(end));
% fprintf('Muon  - Param Error: %.6e ± %.6e\n', mean_errB_muon(end), std_errB_muon(end));
% fprintf('\nMean Total Time:\n');
% fprintf('LSRTR: %.4f seconds\n', mean_time_rgd(end));
% fprintf('Muon:  %.4f seconds\n', mean_time_muon(end));

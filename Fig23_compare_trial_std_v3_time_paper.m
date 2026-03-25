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

n_train = 500;           % # training samples
n_test  = 100;           % # testing samples
sigma   = 0.1;           % y ~ N(mu, sigma^2)

% Algorithm parameters
max_iter     = 40;
alpha        = 0.5;      % RGD stepsize
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
    X = randn(n_train, m);
    mu = X*vec_B_true;
    y_train = mu + sigma * randn(n_train,1);
    
    Xt = randn(n_test, m);
    mut = Xt*vec_B_true;
    y_test = mut + sigma * randn(n_test,1);

    
    
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
                
                vec_grad_B = Omega_kp_sp * (X*vec_B_current - y_train) / n_train;
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
        XB = X * tilde_B_current;
        vec_grad_G = (XB)' * (XB * vec_G_current - y_train) / n_train;
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
                
                vec_grad_B_muon = Omega_kp_sp_muon * (X*vec_B_current_muon - y_train) / n_train;
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
        XB = X * tilde_B_current_muon;
        vec_grad_G_muon = (XB)' * (XB * vec_G_current_muon - y_train) / n_train;
        vec_G_current_muon = vec_G_current_muon - alpha * vec_grad_G_muon;
        G_current_muon = reshape(vec_G_current_muon, rr);
        
        per_iter_muon(iter) = toc(t1);
        
        %% Compute Metrics
        vec_B_current = tilde_B_current * vec_G_current;
        vec_B_current_muon = tilde_B_current_muon * vec_G_current_muon;
        y_predic = Xt * vec_B_current;
        y_predic_muon = Xt * vec_B_current_muon;
        
        loss_history(iter) = norm(X*vec_B_current - y_train)^2/(2*n_train);
        errB_history(iter) = norm(vec_B_current - vec_B_true)^2/norm(vec_B_true)^2;
        erry_history(iter) = norm(y_predic - y_test)^2/norm(y_test)^2;
        
        loss_history_muon(iter) = norm(X*vec_B_current_muon - y_train)^2/(2*n_train);
        errB_history_muon(iter) = norm(vec_B_current_muon - vec_B_true)^2/norm(vec_B_true)^2;
        erry_history_muon(iter) = norm(y_predic_muon - y_test)^2/norm(y_test)^2;
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
% load("LSRTR_Muon_012817_linear_paper_iteration.mat"); 
% figs_path = fullfile('C:\Users\liangx\OneDrive - Iowa State University\Iowa\GLM', 'figure');


fs = 23;

% ===== Figure 1: Loss vs Iterations (log-domain shading, same style as Fig2) =====
figure(1); clf;
hold on;

epsv = 1e-12;

% ---- LSRTR: log-domain mean/std -> back to linear for plotting on log y-axis
L  = log10(all_loss_history + epsv);   % [T x num_trials]
mL = mean(L, 2);
sL = std(L, 0, 2);
mean_loss_plot  = 10.^mL;
lower_loss_plot = 10.^(mL - sL);
upper_loss_plot = 10.^(mL + sL);

fill([iters, fliplr(iters)], [upper_loss_plot; flipud(lower_loss_plot)]', ...
     'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(iters, mean_loss_plot, 'b-', 'LineWidth', 2, 'DisplayName', 'LSRTR');

% ---- Muon
Lm  = log10(all_loss_history_muon + epsv);
mLm = mean(Lm, 2);
sLm = std(Lm, 0, 2);
mean_loss_muon_plot  = 10.^mLm;
lower_loss_muon_plot = 10.^(mLm - sLm);
upper_loss_muon_plot = 10.^(mLm + sLm);

fill([iters, fliplr(iters)], [upper_loss_muon_plot; flipud(lower_loss_muon_plot)]', ...
     'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(iters, mean_loss_muon_plot, 'r-', 'LineWidth', 2, 'DisplayName', 'LSRTR-M');

% ---- SAME formatting as your Figure(2)
xlabel('Number of iterations', 'FontSize', fs);
ylabel('Loss', 'FontSize', fs);
legend('show', 'Location', 'northeast', 'FontSize', fs-3);
set(gcf, 'Color', 'w');
set(gca, 'FontSize', fs);      
set(gca, 'YScale', 'log');
ylim([1e-10, inf]);
axis tight;
% % exportgraphics(gcf, fullfile(figs_path, 'loss_vs_iter.pdf'), 'ContentType', 'vector');



figure(2); clf; hold on;

epsy = 1e-12; 

% ---- LSRTR ----
mu  = mean_errB(:);
sd  = std_errB(:);
logmu = log(mu + epsy);
sdlog = sd ./ (mu + epsy);         
upper = exp(logmu + sdlog);
lower = exp(logmu - sdlog);

fill([iters, fliplr(iters)], [upper; flipud(lower)]', ...
     'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(iters, mu, 'b-', 'LineWidth', 2, 'DisplayName', 'LSRTR');

% ---- MUON ----
mu  = mean_errB_muon(:);
sd  = std_errB_muon(:);
logmu = log(mu + epsy);
sdlog = sd ./ (mu + epsy);
upper = exp(logmu + sdlog);
lower = exp(logmu - sdlog);

fill([iters, fliplr(iters)], [upper; flipud(lower)]', ...
     'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(iters, mu, 'r-', 'LineWidth', 2, 'DisplayName', 'LSRTR-M');

xlabel('Number of iterations', 'FontSize', fs);
% ylabel('$\|\widehat{\underline{\mathbf{B}}}-\underline{\mathbf{B}}\|_F^2/\|\underline{\mathbf{B}}\|_F^2$', ...
%     'Interpreter', 'latex', 'FontSize', fs);
ylabel('$\|\widehat{\mathcal{B}}-\mathcal{B}\|_F^2/\|\mathcal{B}\|_F^2$', ...
    'Interpreter','latex','FontSize',fs);
legend('show', 'Location', 'northeast', 'FontSize', fs-3);
set(gcf, 'Color', 'w');
set(gca, 'FontSize', fs, 'YScale', 'log');
% axis tight; grid on;
% exportgraphics(gcf, fullfile(figs_path, 'estimation_vs_iter_1.pdf'), 'ContentType', 'vector');



% Figure 3: Prediction Error vs Iterations
figure(3); clf;
hold on;
lower_erry = max(mean_erry - std_erry, 1e-10);
upper_erry = mean_erry + std_erry;
fill([iters, fliplr(iters)], [upper_erry; flipud(mean_erry - std_erry)]', ...
     'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(iters, mean_erry, 'b-', 'LineWidth', 2, 'DisplayName', 'LSRTR');
lower_erry_muon = max(mean_erry_muon - std_erry_muon, 1e-10);
upper_erry_muon = mean_erry_muon + std_erry_muon;
fill([iters, fliplr(iters)], [upper_erry_muon; flipud(lower_erry_muon)]', ...
     'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(iters, mean_erry_muon, 'r-', 'LineWidth', 2, 'DisplayName', 'LSRTR-M');
xlabel('Number of iterations', 'FontSize', fs);
ylabel('$\|\widehat{\mathbf{y}}-\mathbf{y}\|_2^2/\|\mathbf{y}\|_2^2$', 'Interpreter', 'latex', 'FontSize', fs);
legend('show', 'Location', 'northeast', 'FontSize', fs-3);
set(gcf, 'Color', 'w');
set(gca, 'FontSize', fs);
set(gca, 'YScale', 'log');
ylim([1e-10, inf]);
axis tight;
% grid on;
%title(sprintf('Prediction Error vs Iterations (%d trials)', num_trials), 'FontSize', fs-2);
% exportgraphics(gcf, fullfile(figs_path, 'prediction_vs_iter_1.pdf'), 'ContentType', 'vector');

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
set(gca, 'YScale', 'log');
axis tight;
% grid on;
%title(sprintf('Loss vs Time (%d trials, mean)', num_trials), 'FontSize', fs-2);
% exportgraphics(gcf, fullfile(figs_path, 'loss_vs_time_1.pdf'), 'ContentType', 'vector');

% Figure 5: Parameter Error vs Time
figure(5); clf;
hold on;
plot(mean_time_rgd, mean_errB, 'b-+', 'LineWidth', 2, ...
     'MarkerIndices', idx_markers, 'MarkerSize', 6, 'DisplayName', 'LSRTR');
plot(mean_time_muon, mean_errB_muon, 'r-*', 'LineWidth', 2, ...
     'MarkerIndices', idx_markers, 'MarkerSize', 6, 'DisplayName', 'LSRTR-M');
xlabel('Running time (s)', 'FontSize', fs);
% ylabel('$\|\widehat{\underline{\mathbf{B}}}-\underline{\mathbf{B}}\|_F^2/\|\underline{\mathbf{B}}\|_F^2$', 'Interpreter', 'latex', 'FontSize', fs);
legend('show', 'Location', 'northeast', 'FontSize', fs-3);
ylabel('$\|\widehat{\mathcal{B}}-\mathcal{B}\|_F^2/\|\mathcal{B}\|_F^2$', ...
    'Interpreter','latex','FontSize',fs);
set(gcf, 'Color', 'w');
set(gca, 'FontSize', fs);
set(gca, 'YScale', 'log');
axis tight;
% grid on;
%title(sprintf('Parameter Error vs Time (%d trials, mean)', num_trials), 'FontSize', fs-2);
% exportgraphics(gcf, fullfile(figs_path, 'estimation_vs_time_1.pdf'), 'ContentType', 'vector');

% Figure 6: Prediction Error vs Time
figure(6); clf;
hold on;
plot(mean_time_rgd, mean_erry, 'b-+', 'LineWidth', 2, ...
     'MarkerIndices', idx_markers, 'MarkerSize', 6, 'DisplayName', 'LSRTR');
plot(mean_time_muon, mean_erry_muon, 'r-*', 'LineWidth', 2, ...
     'MarkerIndices', idx_markers, 'MarkerSize', 6, 'DisplayName', 'LSRTR-M');
xlabel('Running time (s)', 'FontSize', fs);
ylabel('$\|\widehat{\mathbf{y}}-\mathbf{y}\|_2^2/\|\mathbf{y}\|_2^2$', 'Interpreter', 'latex', 'FontSize', fs);
legend('show', 'Location', 'northeast', 'FontSize', fs-3);
set(gcf, 'Color', 'w');
set(gca, 'FontSize', fs);
set(gca, 'YScale', 'log');
axis tight;
% grid on;
%title(sprintf('Prediction Error vs Time (%d trials, mean)', num_trials), 'FontSize', fs-2);
% exportgraphics(gcf, fullfile(figs_path, 'prediction_vs_time_1.pdf'), 'ContentType', 'vector');


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

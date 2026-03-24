% LSR Tensor Linear Regression - Multi-trial varying sample size n
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
n_train_values = [5000, 10000, 15000, 20000, 25000];
num_n_values = length(n_train_values);

n_test  = 5000;           % # testing samples (fixed)


% Algorithm parameters
max_iter     = 30;       % Fixed iterations for each n
alpha        = 0.5;%0.5;      % RGD stepsize
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
        X = randn(n_train, m);
        x_train = X';

        logits = X*vec_B_true;          % n_train x 1
        prob = 1./(1 + exp(-logits));   % sigmoid
        y_train = binornd(1, prob);     

        Xt = randn(n_test, m);
        x_test = Xt';
        logits_t = Xt*vec_B_true;
        prob_t = 1./(1 + exp(-logits_t));
        y_test = binornd(1, prob_t);

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
                    
                    M_kp_sp = Btemp * G_unfold';
                    Omega_kp_sp = zeros(mm(k_prime)*rr(k_prime), n_train);
                    for i = 1:n_train
                        Xi = reshape(X(i,:).', mm);
                        Xi_unf = mode_k_unfold(Xi, k_prime);
                        Otmp = Xi_unf * M_kp_sp;
                        Omega_kp_sp(:, i) = Otmp(:);
                    end
                    
                    logits_current = X*vec_B_current;
                    prob_current = 1./(1 + exp(-logits_current));
                    residual = prob_current - y_train;

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
            vec_B_current = tilde_B_current*vec_G_current;

            XB = X*tilde_B_current;
            logits_current = X*vec_B_current;
            prob_current = 1./(1 + exp(-logits_current));
            residual = prob_current - y_train;

            vec_grad_G = (XB)'*residual/n_train;
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
                    
                    M_kp_sp_muon = Btemp * G_muon_unfold';
                    Omega_kp_sp_muon = zeros(mm(k_prime)*rr(k_prime), n_train);
                    for i = 1:n_train
                        Xi = reshape(X(i,:).', mm);
                        Xi_unf = mode_k_unfold(Xi, k_prime);
                        Otmp = Xi_unf * M_kp_sp_muon;
                        Omega_kp_sp_muon(:, i) = Otmp(:);
                    end
                    
                    logits_muon = X*vec_B_current_muon;
                    prob_muon = 1./(1 + exp(-logits_muon));
                    residual_muon = prob_muon - y_train;

                    vec_grad_B_muon = Omega_kp_sp_muon*residual_muon/n_train;
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

            XB_muon = X*tilde_B_current_muon;
            logits_muon = X*vec_B_current_muon;
            prob_muon = 1./(1 + exp(-logits_muon));
            residual_muon = prob_muon - y_train;

            vec_grad_G_muon = (XB_muon)'*residual_muon/n_train;
            vec_G_current_muon = vec_G_current_muon - alpha * vec_grad_G_muon;
            G_current_muon = reshape(vec_G_current_muon, rr);
        end
        
        %% Store Final Results (after max_iter iterations)
        vec_B_current = tilde_B_current * vec_G_current;
        vec_B_current_muon = tilde_B_current_muon * vec_G_current_muon;

        %LSRTR:err_y
        p_test      = 1./(1+exp(-(Xt*vec_B_current)));
        y_pred = binornd(1, p_test);
        final_erry_rgd(n_idx, trial)      = mean(abs(y_pred  - y_test));

        %LSRTR:loss
        eta_train = X*vec_B_current;
        final_loss_rgd(n_idx, trial) = mean(-y_train.*eta_train + log(1 + exp(eta_train)));
        %LSRTR:err_B
        final_errB_rgd(n_idx, trial) = norm(vec_B_current - vec_B_true)^2/norm(vec_B_true)^2;
        %MUON:Loss
        eta_train_muon = X*vec_B_current_muon;
        final_loss_muon(n_idx, trial) = mean(-y_train.*eta_train_muon + log(1 + exp(eta_train_muon)));
        %MUON:err_B
        final_errB_muon(n_idx, trial) = norm(vec_B_current_muon - vec_B_true)^2/norm(vec_B_true)^2;
        %MUON:err_y
        p_test_muon = 1./(1+exp(-(Xt*vec_B_current_muon)));
        y_pred_muon = binornd(1, p_test_muon);
        final_erry_muon(n_idx, trial) = mean(abs(y_pred_muon - y_test));

    end
end

%% Compute Statistics
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

%% Plotting
% clear; clc;
% load ("compare_trial_std_observation_logistic_120515_paper.mat")
% % save compare_trial_std_observation_logistic_112515.mat
% figs_path = fullfile('C:\Users\liangx\OneDrive - Iowa State University\Iowa\GLM', 'figure');
fs = 20;

% Figure 1: Loss vs Sample Size
figure(4); clf;
hold on;
% LSRTR
lower_loss_rgd = max(mean_loss_rgd - std_loss_rgd, 1e-10);
upper_loss_rgd = mean_loss_rgd + std_loss_rgd;
fill([n_train_values, fliplr(n_train_values)], ...
     [upper_loss_rgd; flipud(lower_loss_rgd)]', ...
     'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(n_train_values, mean_loss_rgd, 'b-o', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'LSRTR');
% Muon
lower_loss_muon = max(mean_loss_muon - std_loss_muon, 1e-10);
upper_loss_muon = mean_loss_muon + std_loss_muon;
fill([n_train_values, fliplr(n_train_values)], ...
     [upper_loss_muon; flipud(lower_loss_muon)]', ...
     'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(n_train_values, mean_loss_muon, 'r-s', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'LSRTR-M');
xlabel('Number of observations', 'FontSize', fs);
ylabel('Loss', 'FontSize', fs);
legend('show', 'Location', 'northeast', 'FontSize', fs-3);
set(gcf, 'Color', 'w');
set(gca, 'FontSize', fs);
set(gca, 'YScale', 'log');
ylim([1e-10, inf]);
axis tight;
grid on;
ax = gca;

ax.XTick = [5000 10000 15000 20000 25000];

ax.XAxis.Exponent = 4;    
% exportgraphics(gcf, fullfile(figs_path, 'loss_vs_n_logistic.pdf'), 'ContentType', 'vector');


% Figure 2: Parameter Error vs Sample Size
figure(5); clf;
hold on;
% LSRTR
lower_errB_rgd = max(mean_errB_rgd - std_errB_rgd, 1e-10);
upper_errB_rgd = mean_errB_rgd + std_errB_rgd;
fill([n_train_values, fliplr(n_train_values)], ...
     [upper_errB_rgd; flipud(lower_errB_rgd)]', ...
     'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(n_train_values, mean_errB_rgd, 'b-o', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'LSRTR');
% Muon
lower_errB_muon = max(mean_errB_muon - std_errB_muon, 1e-10);
upper_errB_muon = mean_errB_muon + std_errB_muon;
fill([n_train_values, fliplr(n_train_values)], ...
     [upper_errB_muon; flipud(lower_errB_muon)]', ...
     'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(n_train_values, mean_errB_muon, 'r-s', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'LSRTR-M');
% xlabel('Number of observations', 'FontSize', fs);
xlabel('Number of observations (\times 10^4)', 'Interpreter','tex', 'FontSize', fs);
ylabel('$\|\widehat{\underline{\mathcal{B}}}-\underline{\mathcal{B}}\|_F^2/\|\underline{\mathcal{B}}\|_F^2$', ...
      'Interpreter', 'latex', 'FontSize', fs);
legend('show', 'Location', 'northeast', 'FontSize', fs-3);
set(gcf, 'Color', 'w');
set(gca, 'FontSize', fs);
set(gca, 'YScale', 'log');
ylim([1e-10, inf]);
axis tight;
grid on;
% ax = gca;                     
% ax.XAxis.Exponent = 0;        
% % ax.XTick = n_train_values;         
% % ax.XTickLabel = compose('%d', n_train_values);
% ax.XTick = [10000 20000];         
% ax.XTickLabel = compose('%d', ax.XTick);
ax = gca;

ax.XTick = [5000 10000 15000 20000 25000];

ax.XAxis.Exponent = 0;    
ax.XTickLabel = compose('%.1f', ax.XTick/1e4);
ax.XTickLabel = strrep(ax.XTickLabel,'.0',''); 

% exportgraphics(gcf, fullfile(figs_path, 'estimation_vs_n_logistic.pdf'), 'ContentType', 'vector');


% Figure 3: Prediction Error vs Sample Size
figure(6); clf;
hold on;
% LSRTR
lower_erry_rgd = max(mean_erry_rgd - std_erry_rgd, 1e-10);
upper_erry_rgd = mean_erry_rgd + std_erry_rgd;
fill([n_train_values, fliplr(n_train_values)], ...
     [upper_erry_rgd; flipud(lower_erry_rgd)]', ...
     'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(n_train_values, mean_erry_rgd, 'b-o', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'LSRTR');
% Muon
lower_erry_muon = max(mean_erry_muon - std_erry_muon, 1e-10);
upper_erry_muon = mean_erry_muon + std_erry_muon;
fill([n_train_values, fliplr(n_train_values)], ...
     [upper_erry_muon; flipud(lower_erry_muon)]', ...
     'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(n_train_values, mean_erry_muon, 'r-s', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'LSRTR-M');
xlabel('Number of observations (\times 10^4)', 'Interpreter','tex', 'FontSize', fs);
% ylabel('$\|\widehat{\mathbf{y}}-\mathbf{y}\|_2^2/\|\mathbf{y}\|_2^2$', 'Interpreter', 'latex', 'FontSize', fs);
ylabel('$\|\hat{y}-y\|_{1}/n_{\mathrm{te}}$', 'Interpreter','latex','FontSize',fs);
legend('show', 'Location', 'northeast', 'FontSize', fs-3);
set(gcf, 'Color', 'w');
set(gca, 'FontSize', fs);
set(gca, 'YScale', 'log');
ylim([1e-10, inf]);
axis tight;
grid on;
% ax = gca;                     
% ax.XAxis.Exponent = 0;        
% % ax.XTick = n_train_values;    
% % ax.XTickLabel = compose('%d', n_train_values);
% ax.XTick = [10000 20000];         
% ax.XTickLabel = compose('%d', ax.XTick);
ax = gca;


ax.XTick = [5000 10000 15000 20000 25000];

ax.XAxis.Exponent = 0;     
ax.XTickLabel = compose('%.1f', ax.XTick/1e4);
ax.XTickLabel = strrep(ax.XTickLabel,'.0','');  
% exportgraphics(gcf, fullfile(figs_path, 'prediction_vs_n_logistic.pdf'), 'ContentType', 'vector');


%% Summary
fprintf('\n========== Summary ==========\n');
fprintf('Sample sizes tested: %s\n', mat2str(n_train_values));
fprintf('Trials per sample size: %d\n', num_trials);
fprintf('Iterations per trial: %d\n\n', max_iter);

fprintf('Results at n = %d:\n', n_train_values(end));
fprintf('LSRTR - Loss: %.6e ± %.6e\n', mean_loss_rgd(end), std_loss_rgd(end));
fprintf('Muon  - Loss: %.6e ± %.6e\n', mean_loss_muon(end), std_loss_muon(end));
fprintf('LSRTR - Param Error: %.6e ± %.6e\n', mean_errB_rgd(end), std_errB_rgd(end));
fprintf('Muon  - Param Error: %.6e ± %.6e\n', mean_errB_muon(end), std_errB_muon(end));
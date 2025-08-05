clc;
clear;
close all;

%% Parameters
n = 1000;
t = linspace(0, 1, n);
numSignals = 5;

%% Step 1: Define Diverse Source Signals
s1 = sin(2*pi*5*t);                       % Sine wave
s2 = square(2*pi*3*t);                    % Square wave
s3 = sawtooth(2*pi*2*t);                  % Sawtooth wave
s4 = sin(2*pi*10*t) .* (t - 0.5);         % Modulated sine (non-linear)
s5 = 2*(rand(1, n) - 0.5);                % Random noise

S = [s1; s2; s3; s4; s5];

% Normalize
S = S - mean(S, 2);
S = S ./ std(S, 0, 2);

%% Step 2: Mix signals
A = randn(numSignals);
X = A * S;

%% Step 3: Whitening (PCA)
X = X - mean(X, 2);
[E, D] = eig(cov(X'));
whitenMat = sqrt(inv(D)) * E';
Z = whitenMat * X;

%% Step 4: ICA using Fixed-Point Iteration with Symmetric Decorrelation

maxIter = 1000;
tol = 1e-6;
W = randn(numSignals);

% Symmetric decorrelation function
sym_decor = @(W) (W * inv(sqrtm(W' * W)));

for iter = 1:maxIter
    W_old = W;
    WX = W * Z;
    g = tanh(WX);
    g_prime = 1 - g.^2;
    W = (g * Z') / n - diag(mean(g_prime, 2)) * W;
    
    W = sym_decor(W); % Symmetric orthogonalization
    
    if norm(abs(diag(W * W_old')) - ones(numSignals,1)) < tol
        break;
    end
end

%% Step 5: Recover Sources
S_est = W * Z;

%% Step 6: Optimal Signal Matching via Correlation Matrix

corrM = abs(corr(S', S_est'));
matching = zeros(1, numSignals);
used = false(1, numSignals);

S_aligned = zeros(size(S_est));

for i = 1:numSignals
    [val, idx] = max(corrM(i,:) .* ~used);
    used(idx) = true;
    sign_corr = sign(corr(S(i,:)', S_est(idx,:)'));
    S_aligned(i,:) = sign_corr * S_est(idx,:) * std(S(i,:)) / std(S_est(idx,:));
end

%% Step 7: Plot Results

% Original Signals
figure('Name', 'Original Signals');
for i = 1:numSignals
    subplot(numSignals,1,i);
    plot(t, S(i,:));
    title(['Original Signal ', num2str(i)]);
    ylabel('Amp'); grid on;
end
xlabel('Time');

% Mixed Signals
figure('Name', 'Mixed Signals');
for i = 1:numSignals
    subplot(numSignals,1,i);
    plot(t, X(i,:));
    title(['Mixed Signal ', num2str(i)]);
    ylabel('Amp'); grid on;
end
xlabel('Time');

% Separated Signals
figure('Name', 'Separated Signals (Aligned)');
for i = 1:numSignals
    subplot(numSignals,1,i);
    plot(t, S_aligned(i,:));
    title(['Recovered Signal ', num2str(i)]);
    ylabel('Amp'); grid on;
end
xlabel('Time');

% Correlation Matrix
figure('Name', 'Correlation Matrix');
imagesc(corr(S', S_aligned'));
colorbar;
title('Correlation Between Original and Recovered Signals');
xlabel('Recovered');
ylabel('Original');
axis square;

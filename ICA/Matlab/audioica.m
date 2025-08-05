clear all;
close all;
clc;
[audio1, fs1] = audioread('Dont start.wav');
[audio2, fs2] = audioread('End of beg (1).wav');

if size(audio1, 2) > 1
    audio1 = mean(audio1, 2);
end
if size(audio2, 2) > 1
    audio2 = mean(audio2, 2);
end
minLen = min(length(audio1), length(audio2));
audio1 = audio1(1:minLen);
audio2 = audio2(1:minLen);
T = minLen;
S = [audio1'; audio2'];  % 2 x T
N = 2; M = 2;

rng(42);
A = rand(M, N);
X = A * S;

X_mean = mean(X, 2);
X_centered = X - X_mean;
covX = cov(X_centered');
[E, D] = eig(covX);
whiteningMatrix = inv(sqrt(D)) * E';
X_whitened = whiteningMatrix * X_centered;

W = zeros(N, N);
for i = 1:N
    w = rand(N, 1);
    w = w / norm(w);

    for iter = 1:500
        w_new = mean((X_whitened .* tanh(w' * X_whitened)), 2) - ...
                mean(1 - tanh(w' * X_whitened).^2) * w;

        if i > 1
            w_new = w_new - W(1:i-1,:)' * (W(1:i-1,:) * w_new);
        end

        w_new = w_new / norm(w_new);
        if norm(w_new - w) < 1e-6
            break;
        end
        w = w_new;
    end
    W(i,:) = w';
end
S_est = W * X_whitened;
S1_est = S_est(1,:) / max(abs(S_est(1,:)));
S2_est = S_est(2,:) / max(abs(S_est(2,:)));

t = (0:T-1)/fs1;

figure('Name','Mixed Signals');
subplot(2,1,1);
plot(t, X(1,:)); title('Mixed Signal 1'); xlabel('Time [s]');
subplot(2,1,2);
plot(t, X(2,:)); title('Mixed Signal 2'); xlabel('Time [s]');

% Play mixed
disp('Playing Mixed Signal 1...');
sound(X(1,:) / max(abs(X(1,:))), fs1);
pause(T/fs1 + 1);
disp('Playing Mixed Signal 2...');
sound(X(2,:) / max(abs(X(2,:))), fs1);
pause(T/fs1 + 1);

% --- Recovered Signal 1 ---
figure('Name','Recovered Signal 1');
plot(t, S1_est, 'r');
title('Recovered Signal 1'); xlabel('Time [s]'); ylabel('Amplitude');
uicontrol('Style', 'pushbutton', ...
          'String', 'Play Signal 1', ...
          'Position', [20 20 120 30], ...
          'Callback', @(src,event)sound(S1_est, fs1));

% --- Recovered Signal 2 ---
figure('Name','Recovered Signal 2');
plot(t, S2_est, 'g');
title('Recovered Signal 2'); xlabel('Time [s]'); ylabel('Amplitude');
uicontrol('Style', 'pushbutton', ...
          'String', 'Play Signal 2', ...
          'Position', [20 20 120 30], ...
          'Callback', @(src,event)sound(S2_est, fs1));

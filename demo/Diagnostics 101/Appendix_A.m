%% Bearing fault simulation signal
% Parameter setting =======================================================
fr = 600;                       % Carrier signal
fd = 13;                        % discrete signal
ff = 10;                        % Characteristic frequency(Modulating signal)
a = 0.02;                       % Damping ratio
T = 1/ff;                       % Cyclic period
fs = 3e3;                       % Sampling rate
K = 50;                         % Number of impulse signal
t = 1/fs:1/fs:2;                % Time
A=5;                            % Maximum amplitude
noise = 0.5;
%==========================================================================

for k = 0 : K-1
    for i = 1 : length(t)
        if t(i)-k*T>=0
            x1(i) = A*exp(-a*2*pi*fr.*(t(i)-k*T));
            x2(i) = sin(2*pi*fr.*(t(i)-k*T));
            x3(i) = x1(i).*x2(i);
        end;end;end
x5 = normrnd(0,noise,1,length(x3));
x4 = 2*sin(2*pi.*t.*fd);
vib = x3 + x4 + x5;
save('Simulation','vib','t')
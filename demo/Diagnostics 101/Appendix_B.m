%=========================== PROBLEM DEFINITION ===========================
rawData = load('Simulation');                                     % Data load
sampRate = 3e3;                                   	   % Sampling rate (Hz)
rpm = 60;                                      		% Shaft rotating speed
bearFreq = [10]*rpm/60;  	% BPFO, BPFI, FTF, BSF
maxP = 300;                                     	  % Maximum order of AR model
windLeng = [2^4 2^5 2^6 2^7];                            % Window length of STFT
%================= Discrete signal separation (AR model) ==================
x=rawData.vib(:); N=length(x);
for p = 1 : maxP  
    if rem(p,50)==0; disp(['p=' num2str(p)]); end
    a = aryule(x,p); % aryule returns the AR model parameter, a(k)
    X = zeros(N,p);
    for i = 1 : p; X(i+1:end,i) = x(1:N-i); end
    xp = -X*a(2:end)';    
    e = x-xp;                                             
    tempKurt(p,1) = kurtosis(e(p+1:end));
end
optP = find(tempKurt==max(tempKurt));                    %==== Optimum solution
optA = aryule(x,optP);
xp = filter([0 -optA(2:end)],1,x);
e = x(optP+1:end) - xp(optP+1:end);                            % residual signal
%================ Demodulation band selection (STFT & SK) =================
Ne = length(e);
numFreq = max(windLeng)+1;
for i = 1 : length(windLeng)
    windFunc = hann(windLeng(i ));          %==== Short Time Fourier Transform
    numOverlap = fix(windLeng(i)/2); 
    numWind = fix((Ne-numOverlap)/(windLeng(i)-numOverlap));                
    n = 1:windLeng(i);
    STFT=zeros(numWind,numFreq);
    for t = 1 : numWind 
        stft = fft(e(n).*windFunc, 2*(numFreq-1));        
        stft = abs(stft(1:numFreq))/windLeng(i)/sqrt(mean(windFunc.^2))*2; 
        STFT(t,:) = stft';
        n = n + (windLeng(i)-numOverlap);
    end    
    for j = 1 : numFreq                                    %==== Spectral Kurtosis
     specKurt(i,j) = mean(abs(STFT(:,j)).^4)./mean(abs(STFT(:,j)).^2).^2-2;
    end
    lgd{i} = ['window size = ',num2str(windLeng(i))];
end                    
figure(1)                                                                %==== Results
freq = (0:numFreq-1)/(2*(numFreq-1))*sampRate;
plot(freq,specKurt); legend(lgd,'location','best')
xlabel('Frequency[Hz]'); ylabel('Spectral kurtosis'); xlim([0 sampRate/2]);
[freqRang] = input('Range of bandpass filtering, [freq1,freq2] = '); 
[b,a] = butter(2,[freqRang(1) freqRang(2)]/(sampRate/2),'bandpass');
X = filter(b,a,e);                                             % band-passed signal
%=========================== Envelope analysis ============================
aX = hilbert(X); % hilbert(x) returns an analytic signal of x
envel = abs(aX);                                          
envel=envel-mean(envel);                                         % envelope signal
fftEnvel = abs(fft(envel))/Ne*2;
fftEnvel = fftEnvel(1:ceil(N/2));
figure(2)                                                           %==== Result plot
freq = (0:Ne-1)/Ne*sampRate;
freq = freq(1:ceil(N/2));
stem(freq,fftEnvel,'LineWidth',1.5); hold on; 
[xx,yy]=meshgrid(bearFreq,ylim);
plot(xx(:,1),yy(:,1),'*-')
% ,xx(:,2),yy(:,2),'x-',xx(:,3),yy(:,3),'d-',xx(:,4),yy(:,4),'^-')
legend('Envelope spectrum','BPFO','BPFI','FTF','BSF');
xlabel('Frequency [Hz]'); ylabel('Amplitude'); xlim([0 max(bearFreq)*1.8])
classdef TimeFrequencyExtractor_alt < Appliable & Uncertainty
    
    properties
        nSegTime = 10;
        nSegFreq = 10;
        dataLength;
        freqLength;
        ind_startTime = [];
        ind_startFreq = [];
    end
    
    methods
        function this = TimeFrequencyExtractor(numSegmentsTime, numSegmentsFreq)
            if nargin > 0
                if exist('numSegmentsTime', 'var') && ~isempty(numSegmentsTime)
                    %if only one argument is given, use the same number of
                    %segments for both domains
                    this.nSegTime = numSegmentsTime;
                    this.nSegFreq = numSegmentsTime;
                end
                if exist('numSegmentsFreq', 'var') && ~isempty(numSegmentsFreq)
                    this.nSegFreq = numSegmentsFreq;
                end
            end
        end
        
        function feat = apply(this, data)
            freqData = abs(fft(data, [], 2));
            freqData = freqData(:, 1:ceil(size(freqData,2)/2));
            
            featTime = cell(this.nSegTime, 1);
            featFreq = cell(this.nSegFreq, 1);

            this.dataLength = size(data,2);
            this.freqLength = size(freqData,2);
   
            for i = 1:this.nSegTime
                %get indices for this segment (Time domain)
                start = (i-1) * floor(size(data,2)/this.nSegTime) + 1;
                stop = min(size(data,2), i * ceil(size(data,2)/this.nSegTime));
                ind = start:stop;
                featTime{i} = this.applyFeatureFuns(data(:,ind));
                this.ind_startTime(1,i) = start;
            end
            for i = 1:this.nSegFreq
                %get indices for this segment (Frequency domain)
                start = (i-1) * floor(size(freqData,2)/this.nSegFreq) + 1;
                stop = min(size(freqData,2), i * ceil(size(freqData,2)/this.nSegFreq));
                ind = start:stop;
                featFreq{i} = this.applyFeatureFuns(freqData(:,ind));
                this.ind_startFreq(1,i) = start;
            end
            
            feat = horzcat(featTime{:}, featFreq{:});
        end

        function infoCell = info(this)
            infoCell = cell(this.nSegTime*9+this.nSegFreq*9,3);
             for i = 1:this.nSegTime
                %get indices for this segment
                start = (i-1) * floor(this.dataLength/this.nSegTime) + 1;
                stop = min(this.dataLength, i * ceil(this.dataLength/this.nSegTime));

                infoCell{9*(i-1)+1,1} = ones(stop-start+1,1);
                infoCell{9*(i-1)+2,1} = ones(stop-start+1,1);
                infoCell{9*(i-1)+3,1} = ones(stop-start+1,1);
                infoCell{9*(i-1)+4,1} = ones(stop-start+1,1);
                infoCell{9*(i-1)+5,1} = ones(stop-start+1,1);
                infoCell{9*(i-1)+6,1} = ones(stop-start+1,1);
                infoCell{9*(i-1)+7,1} = ones(stop-start+1,1);
                infoCell{9*(i-1)+8,1} = ones(stop-start+1,1);
                infoCell{9*(i-1)+9,1} = ones(stop-start+1,1);

                infoCell{9*(i-1)+1,2} = start:stop;
                infoCell{9*(i-1)+2,2} = start:stop;
                infoCell{9*(i-1)+3,2} = start:stop;
                infoCell{9*(i-1)+4,2} = start:stop;
                infoCell{9*(i-1)+5,2} = start:stop;
                infoCell{9*(i-1)+6,2} = start:stop;
                infoCell{9*(i-1)+7,2} = start:stop;
                infoCell{9*(i-1)+8,2} = start:stop;
                infoCell{9*(i-1)+9,2} = start:stop;

                infoCell{9*(i-1)+1,3} = "RMS (Time)";
                infoCell{9*(i-1)+2,3} = "Variance (Time)";
                infoCell{9*(i-1)+3,3} = "Lin. Slope (Time)";
                infoCell{9*(i-1)+4,3} = "Maximum Position (Time)";
                infoCell{9*(i-1)+5,3} = "Maximum (Time)";
                infoCell{9*(i-1)+6,3} = "Minimum (Time)";
                infoCell{9*(i-1)+7,3} = "Skewness (Time)";
                infoCell{9*(i-1)+8,3} = "Kurtosis (Time)";
                infoCell{9*(i-1)+9,3} = "Crest Factor (Time)";
             end

             for i = 1:this.nSegFreq
                %get indices for this segment
                start = (i-1) * floor(this.freqLength/this.nSegFreq) + 1;
                stop = min(this.freqLength, i * ceil(this.freqLength/this.nSegFreq));

                infoCell{9*(i-1)+1+9*this.nSegTime,1} = ones(stop-start+1,1);
                infoCell{9*(i-1)+2+9*this.nSegTime,1} = ones(stop-start+1,1);
                infoCell{9*(i-1)+3+9*this.nSegTime,1} = ones(stop-start+1,1);
                infoCell{9*(i-1)+4+9*this.nSegTime,1} = ones(stop-start+1,1);
                infoCell{9*(i-1)+5+9*this.nSegTime,1} = ones(stop-start+1,1);
                infoCell{9*(i-1)+6+9*this.nSegTime,1} = ones(stop-start+1,1);
                infoCell{9*(i-1)+7+9*this.nSegTime,1} = ones(stop-start+1,1);
                infoCell{9*(i-1)+8+9*this.nSegTime,1} = ones(stop-start+1,1);
                infoCell{9*(i-1)+9+9*this.nSegTime,1} = ones(stop-start+1,1);

                infoCell{9*(i-1)+1+9*this.nSegTime,2} = start:stop;
                infoCell{9*(i-1)+2+9*this.nSegTime,2} = start:stop;
                infoCell{9*(i-1)+3+9*this.nSegTime,2} = start:stop;
                infoCell{9*(i-1)+4+9*this.nSegTime,2} = start:stop;
                infoCell{9*(i-1)+5+9*this.nSegTime,2} = start:stop;
                infoCell{9*(i-1)+6+9*this.nSegTime,2} = start:stop;
                infoCell{9*(i-1)+7+9*this.nSegTime,2} = start:stop;
                infoCell{9*(i-1)+8+9*this.nSegTime,2} = start:stop;
                infoCell{9*(i-1)+9+9*this.nSegTime,2} = start:stop;

                infoCell{9*(i-1)+1+9*this.nSegTime,3} = "RMS (Frequency)";
                infoCell{9*(i-1)+2+9*this.nSegTime,3} = "Variance (Frequency)";
                infoCell{9*(i-1)+3+9*this.nSegTime,3} = "Lin. Slope (Frequency)";
                infoCell{9*(i-1)+4+9*this.nSegTime,3} = "Maximum Position (Frequency)";
                infoCell{9*(i-1)+5+9*this.nSegTime,3} = "Maximum (Frequency)";
                infoCell{9*(i-1)+6+9*this.nSegTime,3} = "Minimum (Frequency)";
                infoCell{9*(i-1)+7+9*this.nSegTime,3} = "Skewness (Frequency)";
                infoCell{9*(i-1)+8+9*this.nSegTime,3} = "Kurtosis (Frequency)";
                infoCell{9*(i-1)+9+9*this.nSegTime,3} = "Crest Factor (Frequency)";
             end
        end
 
        function U = uncertainty(this, U_x, data)
        % Input:    U_x - uncertainty of the sensor, 
        %           data - one sensor
        %           ind_start - start index of each segment
        % Output:   U - uncertainty matrix for means, standarddeviation,
        %           skewness and kurtosis

            U_outTime = [];
            U_outFreq = [];
            % U_x must be a matrix (diagonal matrix means only white noise)
            % If U_x is a full matrix with rowwise uncertainty values for 
            % each cycle (which means only white noise)
            if size(U_x,1) == size(data,1) && size(U_x,2) == size(data,2)
                %do nothing
            % Uncertainty is a row vector, then the row of uncertainty holds 
            % for every row of data
            elseif size(U_x,1) == 1 && size(U_x,2) == size(data,2)
                U_x = U_x.*ones(size(data,1),size(U_x,2));
            % Uncertainty is a upper triangular matrix
            elseif istriu(U_x) && size(U_x,2) == size(data,2)
                U_x = triu(U_x,1) + tril(U_x.');
            % Uncertainty is a lower triangular matrix
            elseif istril(U_x) && size(U_x,2) == size(data,2)
                U_x = triu(U_x.',1) + tril(U_x);
            else
                error('Uncertainty values have wrong dimensions.')
            end
            
            % Save U_x for FFT
            U_store_FFT = TimeFrequencyExtractor_alt.uncertainty_fft(data,U_x);
            U_store_FFT = U_store_FFT(:,1:this.freqLength);
            
            % Save U_x
            U_store = U_x;
            
            % Save data
            data_store = data;
            data_store_fft = abs(fft(data, [], 2));
            data_store_fft = data_store_fft(:, 1:ceil(size(data_store_fft,2)/2));
            
            for i = 1 : size(data_store,1)
                % Uncertainty is a full matrix
                if size(U_store,1) == size(data_store,1) && size(U_store,2) == size(data_store,2)
                    U_x = U_store(i,:).*eye(size(U_store,2));
                    U_xFreq = U_store_FFT(i,:).*eye(size(U_store_FFT,2));
                else
                    error('Uncertainty values have wrong dimensions.')
                end
                
                % Cycle
                data = data_store(i,:);
                data_fft = data_store_fft(i,:);
                
                % Use MC = 1, Do not use MC = 0
                MC = 1;
                
                % Uncertainty for time 
                U_outTime_tmp = TimeFrequencyExtractor_alt.uncertainty_calc(data,this.ind_startTime,U_x,this.nSegTime,this.dataLength,MC);
                U_outTime = [U_outTime; U_outTime_tmp];
                
                % Uncertainty for frequency
                U_outFreq_tmp = TimeFrequencyExtractor_alt.uncertainty_calc(data_fft,this.ind_startFreq,U_xFreq,this.nSegFreq,this.freqLength,MC);
                U_outFreq = [U_outFreq; U_outFreq_tmp];
            end
            
            U = [U_outTime U_outFreq];
        end   
    end
    
    methods(Static)
        function f = applyFeatureFuns(data)
            f = zeros(size(data,1), 9);
            ind = 1:size(data,2);

            % RMS (1)
            f(:,1) = rms(data, 2);
            % Variance (2)
            f(:,2) = var(data, [], 2);
            % Linear Slope (3)
            xm = ind'-mean(ind);
            ym = data(:,ind)-mean(data,2);
            f(:,3) = (ym*xm)./sum(xm.^2);
            % Position of peak (4) and maximum (5)
            [f(:,5), f(:,4)] = max(data, [], 2);
            % Minimum (6)
            f(:,6) = min(data, [], 2);
            % Skewness (7) and kurtosis (8)
            [f(:,7), f(:,8)] = TimeFrequencyExtractor_alt.fastSkewKurt(data);
            % Crest Factor (peak-to-RMS ratio) (9)
            f(:,9) = f(:,5)./f(:,1);
        end
    
        function [s, k] = fastSkewKurt(x)
            dim = 2;
            x0 = x - nanmean(x,dim);
            s2 = nanmean(x0.^2,dim);
            m2 = x0.^2;
            m3 = nanmean(m2.*x0,dim);
            m4 = nanmean(m2.*m2,dim);
            s = m3 ./ s2.^(1.5);
            k = m4 ./ s2.^2;
        end
        
        function U_out = uncertainty_fft(data, U_x)
            U_out = []; 
                        
            % Save U_x
            U_store = U_x;
            
            % Save data
            data_store = data;            
            
            for i = 1 : size(data_store,1)
                % Uncertainty is a full matrix
                if size(U_store,1) == size(data_store,1) && size(U_store,2) == size(data_store,2)
                    U_x = U_store(i,:).*eye(size(U_store,2));
                end
                
                % Cycle
                data = data_store(i,:);
                
                N = size(U_x,2);

                % Matrices with derivatives for real and imaginary parts
                C_cos = zeros(floor(N/2),N);
                C_sin = zeros(floor(N/2),N);
                for k = 0 : floor(N/2)-1 
                    for p = 1 : N
                        beta = 2*pi*(p-1)/N;  
                        C_cos(k+1,p) = cos(k*beta);  
                        C_sin(k+1,p) = -sin(k*beta);
                    end
                end

                % Blockwise calculation for the uncertainty matrices (real /
                % imaginary)
                U_RR = C_cos * U_x * C_cos';
                U_RI = C_cos * U_x * C_sin';
                U_II = C_sin * U_x * C_sin';

                % Matrices with derivatives for amplitude and phase parts
                R = zeros(1,floor(N/2));
                I = zeros(1,floor(N/2));
                A = zeros(1,floor(N/2));
                P = zeros(1,floor(N/2));
                A_R = zeros(1,floor(N/2));
                A_I = zeros(1,floor(N/2));
                P_R = zeros(1,floor(N/2));
                P_I = zeros(1,floor(N/2));
                for k = 0 : floor(N/2)-1 % this.idx
                    tmp_R = 0;
                    tmp_I = 0;
                    for p = 1 : N
                        beta = 2*pi*(p-1)/N;
                        tmp_R = tmp_R + data(1,p) * cos(k*beta);
                        tmp_I = tmp_I + data(1,p) * sin(k*beta); 
                    end
                    R(k+1) = tmp_R;
                    I(k+1) = -tmp_I;
                    A(k+1) = sqrt(R(k+1)^2+I(k+1)^2);
                    P(k+1) = atan(-I(k+1)/R(k+1));
                    A_R(k+1) = R(k+1)/A(k+1);
                    A_I(k+1) = I(k+1)/A(k+1);
                    P_R(k+1) = -I(k+1)/A(k+1)^2;
                    P_I(k+1) = R(k+1)/A(k+1)^2;
                end

                % Blockwise calculation for the uncertainty matrices (amplitude
                % / phase) (U_RR, U_RI, U_II are already uncertainty matrices )
                U11 = diag(A_R) * U_RR * diag(A_R) + diag(A_I) * U_RI' * diag(A_R) ...
                    + diag(A_R) * U_RI * diag(A_I) + diag(A_I) * U_II * diag(A_I);
                U12 = diag(A_R) * U_RR * diag(P_R) + diag(A_I) * U_RI' * diag(P_R) ...
                    + diag(A_R) * U_RI * diag(P_I) + diag(A_I) * U_II * diag(P_I);
                U21 = diag(P_R) * U_RR * diag(A_R) + diag(P_I) * U_RI' * diag(A_R) ...
                    + diag(P_R) * U_RI * diag(A_I) + diag(P_I) * U_II * diag(A_I);
                U22 = diag(P_R) * U_RR * diag(P_R) + diag(P_I) * U_RI' * diag(P_R) ...
                    + diag(P_R) * U_RI * diag(P_I) + diag(P_I) * U_II * diag(P_I);

                % Uncertainty matrix (amplitude / phase)
                U_tmp = [U11 U12; U21 U22];

                % Uncertainty values for the features (as row vector)
                U_tmp = sqrt(abs(diag(U_tmp)))';
                
                U_out = [U_out; U_tmp];
            end 
            U = U_out;        
        end
        
        function U_out = uncertainty_calc(data,ind_start,U_x,nSeg,dataLength,MC)
            U_out = [];
            
            % Calculate stop index of each segment
            ind_stop = [ind_start(2:end)-1, dataLength];

            % Vector for the number of values
            t = 1:size(data,2); 

            % A = RMS, B = var, C = slope, D = peak position, E = max,
            % F = min, G = skewness, H = kurtosis, I = Crest factor
            A = zeros(nSeg,dataLength);
            B = zeros(nSeg,dataLength);
            C = zeros(nSeg,dataLength);
            D = zeros(nSeg,dataLength);
            E = zeros(nSeg,dataLength);
            F = zeros(nSeg,dataLength);
            G = zeros(nSeg,dataLength);
            H = zeros(nSeg,dataLength);
            I = zeros(nSeg,dataLength);

            % Sensitivity coefficient according to GUM
            for i = 1 : nSeg
                % Index of one segment
                ind = ind_start(i) : ind_stop(i);

                % Number of elements in one segment
                N = ind_stop(i)-ind_start(i)+1;

                % Sum (xi-mu)^2
                sum_xi_mu_2 = sum((data(1,ind)-mean(data(1,ind))).^2);

                % Sum (xi-mu)^3
                sum_xi_mu_3 = sum((data(1,ind)-mean(data(1,ind))).^3);

                % Sum (xi-mu)^4
                sum_xi_mu_4 = sum((data(1,ind)-mean(data(1,ind))).^4);
                
                % Sum (xi)^2
                sum_xi_2 = sum(data(1,ind).^2);

                % Mean for one segment
                m = mean(data(1,ind));
                tm = sum(t(ind),2)/size(t(ind),2);

                % Standarddeviation for one segment
                S = std(data(1,ind), [], 2);        

                % Numerator f and denominator g (skewness)
                f = 1/N * sum_xi_mu_3;
                g = (1/N * sum_xi_mu_2)^(3/2);

                % Numerator u and denominator v (kurtosis)
                u = 1/N * sum_xi_mu_4;
                v = (1/N * sum_xi_mu_2)^2;  

                for j = ind
                    % A = RMS, B = var, C = slope, D = peak position, E = max,
                    % F = min, G = skewness, H = kurtosis, I = Crest factor
                    % Sensitivity coefficients (RMS)
                    A(i,j) = data(1,j)/(sqrt(sum_xi_2*N));

                    % Sensitivity coefficients (variance)
                    B(i,j) = 2*((data(1,j)-m)/(N-1));

                    % Sensitivity coefficients (slope)
                    C(i,j) = (t(j)-tm)/(sum((t(j)-tm).^2,2));  
                    
                    % Sensitivity coefficients (position peak, time is
                    % assumed as correct)
                    D(i,j) = 0; % GUM version  

                    % Sensitivity coefficients (max/peak)
                    E(i,j) = 1; % GUM version

                    % Sensitivity coefficients (min)
                    F(i,j) = 1; % GUM version                      

                    % Sensitivity coefficients (skewness) (f1 derivative 
                    % numerator, g1 derivative denominator)
                    f1 = 3/N * ((data(1,j) - m).^2 - 1/N * sum_xi_mu_2);
                    g1 = 3/N * (1/N * sum_xi_mu_2)^(1/2) ...
                        * (data(1,j) - m);
                    G(i,j) = (f1*g-f*g1)/(g.^2);

                    % Sensitivity coefficients (kurtosis) (u1 derivative 
                    % numerator, v1 derivative denominator)
                    u1 = 4/N * ((data(1,j) - m).^3 - 1/N * sum_xi_mu_3);
                    v1 = 4/N^2 * sum_xi_mu_2 * (data(1,j) - m);
                    H(i,j) = (u1*v-u*v1)/(v.^2);

                    % Sensitivity coefficients (Crest Factor)
                    I(i,j) = (-data(1,j)*max(data(1,ind),[],2))/(rms(data(1,ind),2)^3*N); % GUM version  
                end 
            end
                
            % Blockwise calculation of the uncertainty matrix
            U11 = A * U_x * A';
            U12 = A * U_x * B';
            U13 = A * U_x * C';
            U14 = A * U_x * D';
            U15 = A * U_x * E';
            U16 = A * U_x * F';
            U17 = A * U_x * G';
            U18 = A * U_x * H';
            U19 = A * U_x * I';

            U22 = B * U_x * B';
            U23 = B * U_x * C';
            U24 = B * U_x * D';
            U25 = B * U_x * E';
            U26 = B * U_x * F';
            U27 = B * U_x * G';
            U28 = B * U_x * H';
            U29 = B * U_x * I';                

            U33 = C * U_x * C';
            U34 = C * U_x * D';
            U35 = C * U_x * E';
            U36 = C * U_x * F';
            U37 = C * U_x * G';
            U38 = C * U_x * H';
            U39 = C * U_x * I';

            U44 = D * U_x * D';
            U45 = D * U_x * E';
            U46 = D * U_x * F';
            U47 = D * U_x * G';
            U48 = D * U_x * H';
            U49 = D * U_x * I';

            U55 = E * U_x * E';
            U56 = E * U_x * F';
            U57 = E * U_x * G';
            U58 = E * U_x * H';
            U59 = E * U_x * I';

            U66 = F * U_x * F';
            U67 = F * U_x * G';
            U68 = F * U_x * H';
            U69 = F * U_x * I';                

            U77 = G * U_x * G';
            U78 = G * U_x * H';
            U79 = G * U_x * I';

            U88 = H * U_x * H';
            U89 = H * U_x * I';

            U99 = I * U_x * I';                

            % Uncertainty matrix (mean / std / skewness / kurtosis)
            U = [U11 U12 U13 U14 U15 U16 U17 U18 U19; ...
                U12' U22 U23 U24 U25 U26 U27 U28 U29; ...
                U13' U23' U33 U34 U35 U36 U37 U38 U39; ...
                U14' U24' U34' U44 U45 U46 U47 U48 U49; ...
                U15' U25' U35' U45' U55 U56 U57 U58 U59; ...
                U16' U26' U36' U46' U56' U66 U67 U68 U69; ...
                U17' U27' U37' U47' U57' U67' U77 U78 U79; ...
                U18' U28' U38' U48' U58' U68' U78' U88 U89; ...
                U19' U29' U39' U49' U59' U69' U79' U89' U99];

            % Uncertainty values for the features (as row vector)
            U = sqrt(abs(diag(U)))';

            % Sorting according to the features (rms, variance, slope, max pos,
            % max, min, skewness, kurtosis, Crest factor for each segment)
            num_vibs = size(U,2)/9;
            tmp = zeros(size(U,1),size(U,2));
            for j = 1 : 9
                tmp(j:9:end-9+j) = U(1+(j-1)*num_vibs:j*num_vibs);
            end
            U = tmp;
            if MC
                U_MC = TimeFrequencyExtractor_alt.uncertainty_mc(data,ind_start,ind_stop,U_x,nSeg);

                % Overwriting with MC results: MC for max position, max, min, Crest factor
                count = 1;
                for j = [4 5 6 9]
                    U(j:9:end-9+j) = U_MC(1+(count-1)*nSeg:count*nSeg);
                    count = count + 1;
                end
            end
            
            U_out = [U_out; U];
        end
        
        % Monte Carlo method
        function U_out = uncertainty_mc(data,ind_start,ind_stop,U_x,nSeg)
            U_out = [];
            % Runs for Monte Carlo
            runs = 1000000; 
                
            % Number of data points
            n = size(data,2);
                
            % Run MC block-wise 
            blocksize = min(runs,1e3);
            blocks = ceil(runs/blocksize);
            
            % Calculation of sigma
            sigma = sqrt(diag(U_x))';
            
            % Implementation of an update Monte Carlo method
            for m = 1 : blocks
                curr_block = min(blocksize, runs-(m-1)*blocksize);
                Y = zeros(curr_block,4*nSeg);
                for k = 1 : curr_block
                    % Use of normal distributed random numbers
                    dataMC = data + randn(1,n).*sigma;
                    % Calculation of the features for the MC signal
                    for l = 1 : nSeg
                        % Position of peak (nSeg elements) and maximum (nSeg elements)
                        [Y(k,l+nSeg),Y(k,l)] = max(dataMC(ind_start(l):ind_stop(l)), [], 2);
                        % Minimum
                        Y(k,l+2*nSeg) = min(dataMC(ind_start(l):ind_stop(l)), [], 2);
                        % Crest Factor (peak-to-RMS ratio)
                        Y(k,l+3*nSeg) = Y(k,l+nSeg)/rms(dataMC, 2);
                    end   
                end

                if m == 1
                    y = mean(Y);
                    uy = std(Y);
                else
                    % Best estimate (mean) und uncertainty associated with 
                    % the best estimate (std)
                    K = (m-1)*blocksize;
                    K0 = curr_block;
                    % diff to current calculated mean
                    d  = sum( Y - repmat(y,K0,1) )/(K+K0);
                    % new mean
                    y  = y + d;
                    % new variance
                    s2 = ((K-1)*uy.^2 + K*d.^2 + sum( (Y - repmat(y,K0,1)).^2))/(K+K0-1);
                    uy = sqrt(s2);
                end
            end

            % Uncertainty values for the features (as row vector)
            U = uy;
            U_out = [U_out; U];
        end
    end
end
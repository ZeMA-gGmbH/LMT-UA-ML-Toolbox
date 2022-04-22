classdef StatisticalMoments < Appliable & Uncertainty
    %STATISTICAL MOMENTS EXTRACTOR
    
    properties
        nSeg = 10;
        dataLength = [];
        ind_start = [];
    end
    
    methods
        function this = StatisticalMoments(numSegments)
            if nargin > 0
                if exist('numSegments', 'var') && ~isempty(numSegments)
                    this.nSeg = numSegments;
                end
            end
        end
        
        function infoCell = info(this)
            infoCell = cell(this.nSeg*4,3);
             for i = 1:this.nSeg
                %get indices for this segment
                start = (i-1) * ceil(this.dataLength/this.nSeg) + 1;
                stop = min(this.dataLength, i * ceil(this.dataLength/this.nSeg));
                infoCell{4*(i-1)+1,1} = ones(stop-start+1,1);
                infoCell{4*(i-1)+2,1} = ones(stop-start+1,1);
                infoCell{4*(i-1)+3,1} = ones(stop-start+1,1);
                infoCell{4*(i-1)+4,1} = ones(stop-start+1,1);
                infoCell{4*(i-1)+1,2} = start:stop;
                infoCell{4*(i-1)+2,2} = start:stop;
                infoCell{4*(i-1)+3,2} = start:stop;
                infoCell{4*(i-1)+4,2} = start:stop;
                infoCell{4*(i-1)+1,3} = "mean";
                infoCell{4*(i-1)+2,3} = "std";
                infoCell{4*(i-1)+3,3} = "skewness";
                infoCell{4*(i-1)+4,3} = "kurtosis";
             end
        end
        
        function feat = apply(this, data)
            feat = cell(this.nSeg, 1);
            this.ind_start = zeros(1,this.nSeg);
            this.dataLength = size(data,2);
            for i = 1:this.nSeg
                f = zeros(size(data,1), 4);
                
                %get indices for this segment
                start = (i-1) * ceil(size(data,2)/this.nSeg) + 1;
                stop = min(size(data,2), i * ceil(size(data,2)/this.nSeg));
                ind = start:stop;
                
                %compute mean
                f(:,1) = mean(data(:,ind), 2);
                %compute standard deviation
                f(:,2) = std(data(:,ind), [], 2);
                %Remark: The old toolbox structure had skewness and
                %kurtosis switched. This was fixed here but will lead to
                %different feature order.
                %compute skewness
                f(:,3) = skewness(data(:,ind), [], 2);
                %compute kurtosis
                f(:,4) = kurtosis(data(:,ind), [], 2);
                
                feat{i} = f;
                this.ind_start(1,i) = start;
            end
            
            feat = horzcat(feat{:});
        end
        
        function U = uncertainty(this, U_x, data)
        % Input:    U_x - uncertainty of the sensor, 
        %           data - one sensor
        %           ind_start - start index of each segment
        % Output:   U - uncertainty matrix for means, standarddeviation,
        %           skewness and kurtosis

            U_out = []; 
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
            
            % Save U_x
            U_store = U_x;
            
            % Save data
            data_store = data;
            
            for i = 1 : size(data_store,1)
                % Uncertainty is a full matrix
                if size(U_store,1) == size(data_store,1) && size(U_store,2) == size(data_store,2)
                    U_x = U_store(i,:).*eye(size(U_store,2));
                else
                    error('Uncertainty values have wrong dimensions.')
                end
                
                % Cycle
                data = data_store(i,:);
                
                % Calculate stop index of each segment
                ind_stop = [this.ind_start(2:end)-1, this.dataLength];

                % A = mean, B = std, C = skewness, D = kurtosis
                A = zeros(this.nSeg,this.dataLength);
                B = zeros(this.nSeg,this.dataLength);
                C = zeros(this.nSeg,this.dataLength);
                D = zeros(this.nSeg,this.dataLength);

                % Sensitivity coefficient
                for i = 1 : this.nSeg
                    % Index of one segment
                    ind = this.ind_start(i) : ind_stop(i);

                    % Number of elements in one segment
                    N = ind_stop(i)-this.ind_start(i)+1;

                    % Sum (xi-mu)^2
                    sum_xi_mu_2 = sum((data(1,ind)-mean(data(1,ind))).^2);

                    % Sum (xi-mu)^3
                    sum_xi_mu_3 = sum((data(1,ind)-mean(data(1,ind))).^3);

                    % Sum (xi-mu)^4
                    sum_xi_mu_4 = sum((data(1,ind)-mean(data(1,ind))).^4);

                    % Mean for one segment
                    m = mean(data(1,ind));

                    % Standarddeviation for one segment
                    S = std(data(1,ind), [], 2);        

                    % Numerator f and denominator g (skewness)
                    f = 1/N * sum_xi_mu_3;
                    g = (1/N * sum_xi_mu_2)^(3/2);

                    % Numerator u and denominator v (kurtosis)
                    u = 1/N * sum_xi_mu_4;
                    v = (1/N * sum_xi_mu_2)^2;            

                    for j = ind
                        % A = mean, B = std, C = skewness, D = kurtosis
                        % Sensitivity coefficients (mean)
                        A(i,j) = 1/N;

                        % Sensitivity coefficients (std)
                        B(i,j) = (data(1,j)-m)/((N-1)*S);

                        % Sensitivity coefficients (skewness) (f1 derivative 
                        % numerator, g1 derivative denominator)
                        f1 = 3/N * ((data(1,j) - m).^2 - 1/N * sum_xi_mu_2);
                        g1 = 3/N * (1/N * sum_xi_mu_2)^(1/2) ...
                            * (data(1,j) - m);
                        C(i,j) = (f1*g-f*g1)/(g.^2);

                        % Sensitivity coefficients (kurtosis) (u1 derivative 
                        % numerator, v1 derivative denominator)
                        u1 = 4/N * ((data(1,j) - m).^3 - 1/N * sum_xi_mu_3);
                        v1 = 4/N^2 * sum_xi_mu_2 * (data(1,j) - m);
                        D(i,j) = (u1*v-u*v1)/(v.^2);
                    end 
                end

                % Blockwise calculation of the uncertainty matrix
                U11 = A * U_x * A';
                U12 = A * U_x * B';
                U13 = A * U_x * C';
                U14 = A * U_x * D';
                U22 = B * U_x * B';
                U23 = B * U_x * C';
                U24 = B * U_x * D';
                U33 = C * U_x * C';
                U34 = C * U_x * D';
                U44 = D * U_x * D';

                % Uncertainty matrix (mean / std / skewness / kurtosis)
                U = [U11 U12 U13 U14; U12' U22 U23 U24; U13' U23' U33 U34; U14' U24' U34' U44];

                % Uncertainty values for the features (as row vector)
                U = sqrt(abs(diag(U)))';

                % Sorting according to the features (mean, std, skewness,
                % kurtosis for each segment)
                f = size(U,2)/4;
                tmp = zeros(size(U,1),size(U,2));
                for j = 1 : 4
                    tmp(j:4:end-4+j) = U(1+(j-1)*f:j*f);
                end
                U = tmp;
                U_out = [U_out; U];
            end 
            U = U_out;
        end
    end
end


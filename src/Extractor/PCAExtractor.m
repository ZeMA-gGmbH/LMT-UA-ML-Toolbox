classdef PCAExtractor < TransformableFESuperClass & Reconstructor & Uncertainty
    %PCAEXTRACTOR A feature extractor for best PCA coefficients
    %   This is used to extract the best PCA coefficients for the provided
    %   raw data. The coefficients for the principal components of the raw
    %   data are computed and sorted by their explained variance.
    %   The Components that explain the most variance resemble the best.
    %   Features are computed by multiplying the raw data with the sorted
    %   coefficient matrix.
    
    properties
        coeffs = [];     % the PCA coefficients sorted as a result of training
        expl = [];       % variance explained by each principal component
        
        count = 0;
        xiyiSum = [];
        xiSum = [];
        
        heuristic = '';
        numFeat = [];
        
        dsFactor = [];
        theta = [];
        
    end
    
    properties (Constant)
        intendedLength = 500;   % highest number of principal components allowed
    end
    
    methods
        function this = PCAExtractor(varargin)
           if ~isempty(varargin)
               p = inputParser;
               defHeuristic = '';
               expHeuristic = {'elbow','percent'};
               defNumFeat = [];
               addOptional(p,'heuristic',defHeuristic,...
                   @(x) any(validatestring(x,expHeuristic)));
               addOptional(p,'numFeat',defNumFeat,@isnumeric);
               parse(p,varargin{:});
               this.heuristic = p.Results.heuristic;
               this.numFeat = p.Results.numFeat;
           end
        end
        
        
        function infoCell = info(this)
            infoCell = cell(this.numFeat,2);
            for i = 1:size(infoCell,1)
                for lv1 = 1:size(this.coeffs,1)
                    for lv2 = 1:this.dsFactor
                         infoCell{i,1} = [infoCell{i,1} this.coeffs(lv1,i)];
                    end
                end

                infoCell{i,2} = 1:size(this.coeffs,1)*this.dsFactor;
            end
        end
        
        function [this] = trainFromPreTransformed(this,preTransformedData)
            dwnDat = preTransformedData;
            % update summed up covariance matrices
            if isempty(this.xiyiSum)
                this.xiSum = sum(dwnDat,1);
                this.xiyiSum = dwnDat'*dwnDat;
                this.count = size(dwnDat,1);
            else 
                this.xiSum = this.xiSum + sum(dwnDat,1);
                this.xiyiSum = this.xiyiSum + dwnDat'*dwnDat;
                this.count = this.count + size(dwnDat,1);
            end
        end
		
        function dwnDat = pretransform(this, rawData)
            if size(rawData,2) > this.intendedLength
				% downsample raw data for covariance computation
				len = cast(size(rawData,2), 'like', rawData);
				this.dsFactor = cast(round(len/this.intendedLength), 'like', rawData)+1;
                [dwnDat,theta] = resample(rawData', 1, this.dsFactor);
                dwnDat = dwnDat';
                this.theta = theta;
            else
                this.dsFactor = 1;
                this.theta = Inf;
				dwnDat = rawData;
            end
        end
        
        function feat = applyToPretransformed(this, dwnDat)
            this.finishTraining();
            % center the data if it has more than one observation
            if size(dwnDat,1)> 1
                  dwnDat = dwnDat - mean(dwnDat,1);
            end
            if isempty(this.numFeat) && isempty(this.heuristic)
                feat = dwnDat*this.coeffs;
                this.numFeat = size(feat,2);
            elseif isempty(this.heuristic)
                feat = dwnDat*this.coeffs(:,1:this.numFeat);
            elseif strcmp(this.heuristic,'elbow')
                feat = dwnDat*this.coeffs(:,1:FeatureExtractorInterface.elbowPos(this.expl));
                this.numFeat = size(feat,2);
            elseif strcmp(this.heuristic,'percent')
                cutoff = floor(size(this.coeffs,2)/10);
                feat = dwnDat*this.coeffs(:,1:cutoff);
                this.numFeat = size(feat,2);
            end
        end
		
        function this = combine(this, target)
            % combine training results of target with the results of the
            % calling object
            
            % clear previously computed coefficients
            this.trainingFinished = false;
            
            % combine the summed up covariance matrices if classes match
            if strcmp(class(this),class(target))
				if isempty(this.xiSum)
                    this.xiSum = target.xiSum;
                    this.xiyiSum = target.xiyiSum;
                    this.count = target.count;
				else
                    this.xiSum = this.xiSum + target.xiSum;
                    this.xiyiSum = this.xiyiSum + target.xiyiSum;
                    this.count = this.count + target.count;
				end
            else
                warning(['Classes ',class(this),' and ',class(target),...
                    ' do not match and cannot be combined']);
            end
        end
        
        
        function rec = reconstruct(this, feat)
            rec = zeros(size(feat,1), size(this.coeffs,2));
            for i = 1:size(feat, 2)
                rec = rec + feat(:,i) .* this.coeffs(:,i)';
            end
            % revert previous centering if there is more than one
            % observation
            if size(feat,1)> 1
                rec = rec + (this.xiSum/this.count);
            end
            %upsample, if neccesary
            rec = resample(rec', this.dsFactor, 1)';
        end
        
        
        function U = uncertainty(this, U_x, data)
        % Input:    U_x - uncertainty of the sensor, 
        %           data - one sensor
        % Output:   U - uncertainty matrix for wavelet coefficients
        %           (approximation coefficients of highest level, all 
        %           detail coefficients)
            theta = this.theta;
            dsFactor = this.dsFactor;
            org_data = data;
            data = resample(data', 1, dsFactor)';
            U_out = [];
            
            % U_x must be a matrix (diagonal matrix means only white noise)
            % If U_x is a full matrix with rowwise uncertainty values for 
            % each cycle (which means only white noise)
            if size(U_x,1) == size(org_data,1) && size(U_x,2) == size(org_data,2)
                %do nothing
            % Uncertainty is a row vector, then the row of uncertainty holds 
            % for every row of data
            elseif size(U_x,1) == 1 && size(U_x,2) == size(org_data,2)
                U_x = U_x.*ones(size(data,1),size(U_x,2));
            % Uncertainty is a upper triangular matrix
            elseif istriu(U_x) && size(U_x,2) == size(org_data,2)
                U_x = triu(U_x,1) + tril(U_x.');
            % Uncertainty is a lower triangular matrix
            elseif istril(U_x) && size(U_x,2) == size(org_data,2)
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
                if size(U_store,1) == size(org_data,1) && size(U_store,2) == size(org_data,2)
                    U_x = U_store(i,:).*eye(size(U_store,2));
                else
                    error('Uncertainty values have wrong dimensions.')
                end
                
                % One cylce
                data_cyc = data(i,:);
                
                % Number of data points
                n = size(data_cyc,2);

                % Calculation of sigma
                sigma_tmp = sqrt(diag(U_x))';
                
                % FIR filter function if a downsampling is done
                if theta ~= Inf 
                    
                    % FIR filter size                
                    Ntheta = length(theta);
                    
                    % Extend sigma
                    sigma_tmp = [sigma_tmp, zeros(1,Ntheta-1)];
                    
                    % Sigma must be squared
                    sigma2 = sigma_tmp.^2;
                    
                    % V needs to be extended to cover Ntheta timesteps more into the past
                    sigma2_extended = [zeros(1,Ntheta-1) sigma2];
                    V = diag(sigma2_extended);
                    
                    % Extended signal
                    xext = [0,org_data(i,:),zeros(1,Ntheta)]; 

                    % apply FIR filter to calculate best estimate in accordance with GUM
                    Lhalf = ceil(length(theta)/2);      % half of filter length
                    tmp = xext(1)*ones(1,length(theta));% const. signal
                    [~,zi] = filter(theta,1.0,tmp);     % calculate steady state
                    x = filter(theta,1.0,xext,zi);      % FIR filter output signal   
                    x = x(2:end);                       % remove 0 at the beginning
                    x_down_tmp = x(Lhalf:dsFactor:end);     % downsample
                    x_down = x_down_tmp(1:end-(size(x_down_tmp,2)-n));          % remove values at the end
                    
                    % UncCov needs to be calculated inside in its own for-loop
                    % V has dimension (length(sigma2) + Ntheta) * (length(sigma2) + Ntheta) 
                    % --> slice a fitting Ulow of dimension (Ntheta x Ntheta)
                    UncCov = zeros(1,length(sigma2));
                    for k = 1 : length(sigma2)
                        Ulow = V(k:k+Ntheta-1,k:k+Ntheta-1);
                        % Static part of uncertainty
                        UncCov(1,k) = flip(theta)*(Ulow*flip(theta)'); 
                    end

                    % point-wise standard uncertainties associated with x (wrt downsampling)   
                    ux = sqrt(abs(UncCov));
                    sigma = ux(Lhalf:dsFactor:end);
                    sigma = sigma(1:size(data,2));
                else
                    sigma = sigma_tmp;
                end

                % Implementation of an update Monte Carlo method
                runs = 1000;

                % run MC block-wise 
                blocksize = min(runs,1e3);
                blocks = ceil(runs/blocksize);

                for m = 1 : blocks
                    curr_block = min(blocksize, runs-(m-1)*blocksize);
                    Y = zeros(curr_block,n);
                    parfor k = 1 : curr_block
                        % Use of normal distributed random numbers
                        dataMC = data_cyc + randn(1,n).*sigma;
                        % Calculation of the features for the MC signal
                        Y(k,:) = dataMC*this.coeffs;   
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
            U = U_out;   
        end
        
    end
    methods (Access = protected)
        function finishTraining(this)
            % compute pca of the summed up covariance matrices
            covariance = 1/(this.count-1) * (this.xiyiSum - (1/this.count)*this.xiSum'*this.xiSum);
            [coeff,~,explained] = pcacov(covariance);
            
            % save computed coefficients and explained curve
            this.coeffs = coeff;
            this.expl = explained;
            
            this.trainingFinished = true;
        end
    end
end


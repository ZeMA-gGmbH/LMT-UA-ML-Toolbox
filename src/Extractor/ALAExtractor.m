classdef ALAExtractor < TransformableFESuperClass & Reconstructor & Uncertainty
    %ALAEXTRACTOR A feature extractor for ALA features
    %   This is used to extract features via the ALA method. Therefore
    %   cycles are divided into several intervals. Slope and mean for each
    %   interval are then combined to form the extracted features.
    
    properties
        errVec = [];
		l = [];
		start = [];
		stop = [];
        dsFactor = [];
        
        numFeat = [];
        
        theta = [];
    end
    
    properties (Constant)
        intendedLength = 500;   % highest number of principal components allowed
    end
    
    methods
        function this = ALAExtractor(varargin)
           if ~isempty(varargin)
               p = inputParser;
               defNumFeat = [];
               addOptional(p,'numFeat',defNumFeat,@isnumeric);
               parse(p,varargin{:});
               this.numFeat = p.Results.numFeat;
           end
        end
        

        
        function infoCell = info(this)
            infoCell = cell(this.numFeat,3);
            
            for i = 1:this.numFeat/2

                if(this.start(i) == 1)
                    infoCell{2*i-1,2} = [this.start(i):this.dsFactor*this.stop(i)];
                    infoCell{2*i,2} = [this.start(i):this.dsFactor*this.stop(i)];
                else
                    infoCell{2*i-1,2} = [this.dsFactor*this.start(i):this.dsFactor*this.stop(i)];
                    infoCell{2*i,2} = [this.dsFactor*this.start(i):this.dsFactor*this.stop(i)];
                end
                infoCell{2*i-1,1} = ones(1,size(infoCell{2*i-1,2},2));
                infoCell{2*i,1} = ones(1,size(infoCell{2*i-1,2},2));
                infoCell{2*i-1,3} = ["Mean"];
                infoCell{2*i,3} = ["Slope"];
            end
        end
        
		function [this] = trainFromPreTransformed(this,preTransformedData)
            % clear previously computed coefficients
            this.start = [];
            this.stop = [];
            
            dwnDat = preTransformedData;
			%compute error matrix
			for i = 1:size(dwnDat,1)
				if i == 1
					errVec = this.errMatTransformFast_mex(dwnDat(i,:));
				else
					errVec = errVec + this.errMatTransformFast_mex(dwnDat(i,:));
				end
			end
            
			this.l = size(dwnDat,2);
			
            % update summed up error vector
            if isempty(this.errVec)
                this.errVec = errVec;
            else 
                this.errVec = this.errVec + errVec;
            end
        end
		
        function dwnDat = pretransform(this, rawData)
			if size(rawData,2) > this.intendedLength
				% downsample raw data for covariance computation
				len = cast(size(rawData,2), 'like', rawData);
				this.dsFactor = cast(round(len/this.intendedLength), 'like', rawData)+1; %bei alten Ergebnissen muss +1 weg, damit es geht!
				[dwnDat,theta] = resample(rawData', 1, this.dsFactor);
                dwnDat = dwnDat';
                this.theta = theta;
			else
				dwnDat = rawData;
                this.dsFactor = 1;
                this.theta = Inf;
            end
        end
        
        function fitParam = applyToPretransformed(this, dwnDat)
            this.finishTraining();
			%Compute linear fit parameter
			fitParam = zeros(size(dwnDat,1), length(this.start)*2, 'like', dwnDat);
			x = 1:cast(length(dwnDat), 'like', dwnDat);
			for i = 1:cast(length(this.start), 'like', dwnDat)
				ind = this.start(i):this.stop(i);
				for n = 1:size(dwnDat,1)
					[~, d] =  this.linFit(x(ind),dwnDat(n, ind));
					fitParam(n,[2*i-1,2*i]) = d;
				end
            end
        end
		
		function this = combine(this, other)
			this.start = [];
			this.stop = [];
            this.trainingFinished = false;
			
			if isempty(this.errVec)
				this.errVec = other.errVec;
				this.l = other.l;
			else
				this.errVec = this.errVec + other.errVec;
			end
        end
        
        function reconstruction = reconstruct(this, feat)
            %reconstruct downsampled data from linear fits
            rec = zeros(size(feat,1), this.stop(end));
            for i = 1:length(this.start)
                ind = this.start(i):this.stop(i);
                len = this.stop(i)-this.start(i)+1;
                meanInd = ((i-1)*2)+1;
                slopeInd = meanInd +1;
                nSamples = size(feat,1);
                
                rec(:, ind) = repmat(feat(:, meanInd), 1, len);
                slopeAdd = repmat(ind, nSamples, 1) .* feat(:,slopeInd);
                rec(:, ind) = rec(:, ind) + slopeAdd - mean(slopeAdd, 2);
            end
            
            %upsample, if neccesary
            reconstruction = resample(rec', this.dsFactor, 1)';
        end
        
        function U = uncertainty(this, U_x, data)
        % Input:    U_x - uncertainty of the sensor, 
        %           data - one sensor
        % Output:   U - uncertainty matrix for mean and slope coefficients
        
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
                
                % Cycle
                data = data_store(i,:);
                
                % Number of data points
                n = size(data,2);

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
                    for k = 1  : length(sigma2)
                        Ulow = V(k:k+Ntheta-1,k:k+Ntheta-1);
                        % Static part of uncertainty
                        UncCov(1,k) = flip(theta)*(Ulow*flip(theta)'); 
                    end

                    % point-wise standard uncertainties associated with x (wrt downsampling)   
                    ux = sqrt(abs(UncCov));
                    sigma = ux(Lhalf:dsFactor:end);
                    sigma = sigma(1:size(data,2));            
                    U_x = sigma.^2.*eye(size(data,2)); 
                end                
                
                % Vector of all splitpoints including the first and last one
                v = [this.start size(data,2)];

                % Number of sections, length of one cylce
                num_sections = length(this.start);
                l_cycle = this.l;

                % Vector for the number of values
                t = 1:size(data,2); 

                % Calculate mean of time and data for every section
                % k-th column represents the k-th section
                tm = zeros(1,num_sections);
                ym = zeros(1,num_sections);
                for k = 1 : num_sections
                    % tm is independet of data (i.e. the same for every cycle)
                    tm(1,k) = sum(t(v(k):v(k+1)),2)/size(t(v(k):v(k+1)),2);
                    ym(1,k) = sum(data(1,v(k):v(k+1)),2)/size(data(1,v(k):v(k+1)),2);
                end

                % Calculate Jacobian matrix for one cycle
                % Begin of new section split point + number of sections - 1
                diff_yk_yj = zeros(1,l_cycle+num_sections-1);
                diff_bk_yj = zeros(1,l_cycle+num_sections-1);
                for k = 1 : num_sections
                    ind_start = v(k) + k - 1;
                    ind_stop = v(k+1) + k - 1;
                    tmp = 1/(v(k+1)-v(k)+1);
                    % Sensitivity coefficient for the mean yk (derivative yk 
                    % with respect to yj, independent of yj and tj)
                    diff_yk_yj(1,ind_start:ind_stop) = tmp;
                    % Sensitivity coefficient for the slope bk (derivative bk 
                    % wrt yj, independent of yj)
                    diff_bk_yj(1,ind_start:ind_stop) = (t(v(k):v(k+1))-tm(1,k))...
                        /(sum((t(v(k):v(k+1))-tm(1,k)).^2,2));
                end

                % Sensitivity coefficient of the means:  C
                % Sensitivity coefficient of the slopes: D
                C = zeros(size(v,2)-1,l_cycle);
                D = zeros(size(v,2)-1,l_cycle);
                for i = 1 : num_sections
                    % Sensitivity coefficient of the means
                    C(i,v(i):v(i+1)) = diff_yk_yj(:,v(i)+i-1:v(i+1)+i-1);
                    % Sensitivity coefficient of the slopes
                    D(i,v(i):v(i+1)) = diff_bk_yj(:,v(i)+i-1:v(i+1)+i-1);
                end

                % Blockwise calculation of the uncertainty matrix
                U11 = C * U_x * C';
                U12 = C * U_x * D';
                U22 = D * U_x * D';

                % Uncertainty matrix
                U = [U11 U12; U12' U22];

                % Uncertainty values for the features (as row vector)
                U = sqrt(abs(diag(U)))';

                % Sorting according to the features (mean, slope for each segment)
                f = size(U,2)/2;
                tmp = zeros(size(U,1),size(U,2));
                for j = 1 : 2
                    tmp(j:2:end-2+j) = U(1+(j-1)*f:j*f);
                end
                U = tmp;
                U_out = [U_out; U];
            end 
            U = U_out;
        end
    end
	
	methods (Access = protected)
		function finishTraining(this)
			
			errMat = Inf(this.l, 'like', this.errVec);
			%from-to matrix
			errMat(tril(true(this.l),-1)) = this.errVec;
            errMat = errMat';
			
			if isa(errMat, 'gpuArray')
                errMat = gather(errMat);
            end
			if isempty(this.numFeat)
                [~, splits, e] = this.findSplits(errMat);
            elseif this.numFeat >=4
                numSplits = floor(this.numFeat/2)-1;
                [~, splits, e] = this.findSplits(errMat,numSplits);
            else
                disp('specifiy higher numFeat so splitting is relevant')
            end
            this.start = cast([1,splits], 'like', this.errVec);
            this.stop = cast([splits, this.l], 'like', this.errVec);
            
            if any(isinf(this.start)) || any(isinf(this.stop))
                error('Failed to find linear segments');
            end
            this.numFeat = size(this.start,2)*2;
            this.trainingFinished = true;
		end
	end
	
	methods(Static)
		%ToDo: Umschreiben für Matrizen
		function errMat = errMatTransformFast_mex( data )
			%ERRMATTRANSFORMFAST Summary of this function goes here
			%   Detailed explanation goes here
			len = length(data);
			errMat = zeros(1, (len*(len-1)/2));
			indRunning = 1;
			%iterate over start-points
			for i = 1:len
				sumX = i;
				sumXX = i^2;
				sumY = data(i);
				sumYY = data(i)^2;
				sumXY = i * data(i);
				%iterate over stop-points
				for j = i+1:len
					sumX = sumX + j;
					sumXX = sumXX + j^2;
					sumY = sumY + data(j);
					sumYY = sumYY + data(j)^2;
					sumXY = sumXY + j*data(j);
					num = j-i+1;
					f = -1/num;
					
					p1 = sumXX - sumX^2/num;
					p2 = 2*sumX*sumY/num - 2*sumXY;
					p3 = sumYY - sumY^2/num;
					b = (sumXY - sumX*sumY/num)/(sumXX - sumX^2/num);
					errMat(indRunning) = p1*b^2+p2*b+p3;
					
					indRunning = indRunning + 1;
				end
			end
        end        
    end
    
    methods(Access = private)
        function [R2, data] = linFit(~, x, y)
            xm = sum(x,2)/size(x,2);
            ym = sum(y,2)/size(y,2);
            xDiff = (x-repmat(xm,1,size(x,2)));
            b = sum((xDiff).*(y-repmat(ym,1,size(y,2))), 2)./sum((xDiff).^2,2);
            a = ym - b.*xm;
            R2 = 0;
            for i = 1:size(y,1)
                R2 = R2 + sum((y(i,:) - (a(i) + b(i) * x(i,:))).^2);
            end
            data = [ym,b];
        end
		
        function [ err, splits, dat ] = findSplits( this, errMat, numSplits )
        %FINDSPLITS Summary of this function goes here
        %   Detailed explanation goes here
            maxSplits = 70;
            n = length(errMat);
            spl = Inf(maxSplits,n);
            errors = Inf(maxSplits,n); %#ok<PROPLC>
            for q = 1:maxSplits
                for i = 1:n-q
                    if q == 1
                        sumRes = errMat(i,:) + errMat(:,n)';
                    else
                        sumRes = errors(q-1,:) + errMat(i,:); %#ok<PROPLC>
                    end
                    [errors(q,i),spl(q,i)] = min(sumRes); %#ok<PROPLC>
                end
            end

            dat = errors(:,1)'; %#ok<PROPLC>
            sqErr = this.getFitErrorMatrix(dat, {'this.linFit'});
            maxSplits = 3;
            n = length(sqErr);
            splTemp = Inf(maxSplits,n);
            errorsTemp = Inf(maxSplits,n);
            for q = 1:maxSplits
                for i = 1:n-q
                    if q == 1
                        sumRes = sqErr(i,:) + sqErr(:,n)';
                    else
                        sumRes = errorsTemp(q-1,:) + sqErr(i,:);
                    end
                    [errorsTemp(q,i),splTemp(q,i)] = min(sumRes);
                end
            end
            splits = zeros(1,maxSplits);
            splits(1) = splTemp(maxSplits,1);
            for i = maxSplits-1:-1:1
                splits(maxSplits - i + 1) = splTemp(i, splits(maxSplits - i));
            end

            if nargin < 3
                numSplits = splits(end);
            end
            splits = zeros(1,numSplits);
            err = errors(numSplits,1); %#ok<PROPLC>
            splits(1) = spl(numSplits,1);
            for i = numSplits-1:-1:1
                splits(numSplits - i + 1) = spl(i, splits(numSplits - i));
            end
        end
		
		function [sqErr, functions] = getFitErrorMatrix(this, dat, varargin)
            dat = sum(dat,1);
            x = 1:size(dat,2);

            fitFunctions = varargin{1};

            N = size(dat,2);
            sqErr = Inf(N);
            functions = cell(N);

            combos = this.nchoose2(1:N);
            sqErrTemp = Inf(1,size(combos,1));
            funTemp = zeros(1,size(combos,1));
            for i = 1:size(combos,1)
                sqTT = Inf(1,length(fitFunctions));
                for j = 1:length(fitFunctions)
                    for k = 1:size(dat,1)
                        xu = x(combos(i,1):combos(i,2));
                        yu = dat(k,combos(i,1):combos(i,2));
                        err = this.linFit(xu, yu);
                        if sqTT(j) == Inf
                            sqTT(j) = err;
                        else
                            sqTT(j) = sqTT(j) + err;
                        end
                    end
                end
                [sqErrTemp(i), funTemp(i)] = min(sqTT);
            end
            for i = 1:size(combos,1)
                functions(combos(i,1),combos(i,2)) = fitFunctions(funTemp(i));
                sqErr(combos(i,1),combos(i,2)) = sqErrTemp(i);
            end
        end
		
        function [ combos ] = nchoose2( ~, nums,varargin )
        %FNCHOOSEK Summary of this function goes here
        %   Detailed explanation goes here
            N = length(nums);
            combos = zeros(nchoosek(N,2),2);
            for i = 1:N-1
                start = nchoosek(N, 2) - nchoosek(N - i + 1, 2) + 1; %#ok<PROPLC>
                if N-i == 1
                    fin = length(combos);
                else
                    fin = nchoosek(N, 2) - nchoosek(N-i, 2);
                end
                combos(start:fin, 1) = nums(i); %#ok<PROPLC>
                combos(start:fin, 2) = nums(i+1:end); %#ok<PROPLC>
            end
        end
	end
    
end

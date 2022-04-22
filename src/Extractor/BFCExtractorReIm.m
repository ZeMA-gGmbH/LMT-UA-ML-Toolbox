classdef BFCExtractorReIm < TransformableFESuperClass & Reconstructor & Uncertainty
    %BFCEXTRACTOR 

    properties
        m = [];
        n = [];
        ind = [];
        idx = []; % sorted index
        
        heuristic = '';
        numFeat = [];
     end
    
    methods
        function this = BFCExtractor(varargin)
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
            infoCell = cell(this.numFeat*2,3);
            counter = 1;
            for i = 1:size(this.ind,2)

                if(this.ind(i) == 1)
                    infoCell{counter,3} = ["ABS FFT Component"];
                    infoCell{counter,2} = [i-1];
                    infoCell{counter,1} = [1];
                    infoCell{counter+this.numFeat,3} = ["Phase FFT Component"];
                    infoCell{counter+this.numFeat,2} = [i-1];
                    infoCell{counter+this.numFeat,1} = [1];
                    counter = counter+1;
                end
            end
        end
        
        function this = trainFromPreTransformed(this, preTransformedData)
			this.ind = [];
            
            amp = preTransformedData;
           
            if isempty(this.m)
                this.m = sum(amp,1);
                this.n = size(amp,1);
            else
                this.m = this.m + sum(amp,1);
                this.n = this.n + size(amp, 1);
            end
            %clear preTransformedData;
        end
		
		function preTransformed = pretransform(this, rawData)
			coeff = fft(rawData, [], 2);
            preTransformed = coeff(:, 1:floor(size(rawData,2)/2));
		end
		
		function features = applyToPretransformed(this, preTransformed)
            finishTraining(this);
            coeff = preTransformed(:,this.ind);
            features = [real(coeff), imag(coeff)];
		end
        
        function obj1 = combine(this, obj1, obj2)
			this.trainingFinished = false;
            obj1.ind = [];
            if ~isempty(obj1.m)
                obj1.m = obj1.m + obj2.m;
                obj1.n = obj1.n + obj2.n;
            else
                obj1.m = obj2.m;
                obj1.n = obj2.n;
            end
        end
        
        function rec = reconstruct(this, feat)
            re = feat(:, 1:size(feat,2)/2);
            im = feat(:, size(feat,2)/2+1:end);
            
            fcoeff = complex(zeros(size(feat,1), length(this.ind)));
            fcoeff(:,this.ind) = complex(re, im);
            fcoeff = [fcoeff, flip(fcoeff,2)];
            
            rec = ifft(fcoeff, [], 2, 'symmetric');
        end
        
        function U = uncertainty(this, U_x, data)
        % Input:    U_x - uncertainty of the sensor, 
        %           data - one sensor
        % Output:   U - uncertainty matrix for amplitude / phase
            
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

                % Uncertainty matrix (real / imaginary)
                U_tmp = [U_RR U_RI; U_RI' U_II]; 

                % Uncertainty values for the features (as row vector)
                U_tmp = sqrt(abs(diag(U_tmp)))';

                % Index of relevant features
                index = this.idx(1:this.numFeat);
                max_num_feat = size(this.ind,2);

                % Uncertainty for relevant features
                U = horzcat(U_tmp(index),U_tmp(index+max_num_feat));
                U_out = [U_out; U];
            end 
            U = U_out;
         end
    end
    
    methods (Access = protected)
        function finishTraining(this)
            mean = this.m ./ this.n;
            [mean, idx] = sort(mean, 'descend');
            i = false(size(mean));
            if isempty(this.numFeat) && isempty(this.heuristic)
                nFeat = floor(size(mean, 2)/10);
            elseif isempty(this.heuristic)
                nFeat = this.numFeat;
            elseif strcmp(this.heuristic,'elbow')
                nFeat = FeatureExtractorInterface.elbowPos(mean);
            end
            if(this.trainingFinished == false)
                this.numFeat = nFeat;
            end
            i(idx(1:nFeat)) = true;
            this.ind = i;
			this.trainingFinished = true;
            this.idx = idx;
        end
    end
end


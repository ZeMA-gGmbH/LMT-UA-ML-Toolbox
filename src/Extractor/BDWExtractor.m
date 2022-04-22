classdef BDWExtractor < TransformableFESuperClass & Reconstructor & Uncertainty
    %BDWEXTRACTOR A feature extractor for best daubechies wavelet coefficients (BDW)
    %   This is used to extract the best daubechies wavelet coefficients
    %   for the provided raw data.
    
    properties
        ind = [];%the indices ordering the wavelet coefficients that result from training
        m = [];  %the sum of wavelet coefficients for the ongoing training 
        n = [];  %counter
        heuristic = '';
        numFeat = [];
        idx = []; % sorted index
        
        len = [];
        originLen = [];
    end
    
    methods
        function this = BDWExtractor(varargin)
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
            infoCell = cell(this.numFeat,3);
            counter = 1;
            for i = 1:size(this.ind,2)

                if(this.ind(i) == 1)
                    infoCell{counter,3} = ["BDW Coeffizient"];
                    infoCell{counter,2} = [i];
                    infoCell{counter,1} = [1];
                    counter = counter+1;
                end
            end
        end
        
        function [this] = trainFromPreTransformed(this, preTransformedData)
            this.ind = [];
            data = preTransformedData;
            
            % sum up all wavelet coefficients and store the result for
            % continued training
            if isempty(this.m)
                this.m = sum(abs(data));
                this.n = size(data,1);
            else
                this.m = this.m + sum(abs(data));
                this.n = this.n + size(data,1);
            end
        end
        
        function data = pretransform(this, data)
            this.originLen = size(data, 2);
            % compute wavelet transformation
            wlevel = BDWExtractor.wMaxLv(size(data,2));
            [af, df] = wfilters('db2');

            d = cell(1,wlevel);
            for i = wlevel:-1:1
                l = size(data,2)+6;
                fInd = false(1,l);
                fInd(5:2:l) = true;
                data = [data(:,[1,1,1]), data, data(:,[end, end, end])];
                d{i} = filter(df, 1, data, [], 2);
                d{i} = d{i}(:,fInd);

                data = filter(af, 1, data, [], 2);
                data = data(:,fInd);
            end
            % concatenate coefficients of different levels
            this.len = [size(data,2), cellfun(@(a)size(a,2), d)];
            data = [data, d{:}];
        end
        
        function feat = applyToPretransformed(this, data)
            finishTraining(this);
            feat = data(:,this.ind);
        end
        
        
        function this = combine(this, target)
            % combine training results of target with the results of the
            % calling object
            
            % clear previously computed coefficient order
            this.ind = [];
            this.trainingFinished = false;
            
            % combine the summed up coefficients if classes match
            if strcmp(class(this),class(target))
                if isempty(this.m)
                    this.m = target.m;
                    this.n = target.n;
                else
                    this.m = this.m + target.m;
                    this.n = this.n + target.n;
                end
            else
                warning(['Classes ',class(this),' and ',class(target),...
                    ' do not match and cannot be combined']);
            end
        end
        
        function rec = reconstruct(this, feat)
            [~,~, LoR, HiR] = wfilters('db2');
            
            coeff = zeros(size(feat,1), sum(this.len));
            coeff(:,this.ind) = feat;
            
            %ToDo: remove this block of code and figure out filter delays
            %below!
            rec = zeros(size(feat,1), this.originLen);
            for i = 1:size(feat,1)
                rec(i,:) = waverec(coeff(i,:), [this.len this.originLen], 'db2');
            end
            
%             clen = [0 cumsum(this.len)];
%             d = cell(length(this.len), 1);
%             for i = 1:length(this.len)
%                 d{i} = coeff(:, clen(i)+1:clen(i+1));
%             end
            
%             rec = d{1};
%             for i = 2:length(this.len)
%                 rec = filter(LoR, 1, upsample(rec', 2)', [], 2);
%                 detail = filter(HiR, 1, upsample(d{i}', 2)', [], 2);
%                 detail(size(feat,1), size(rec,2)) = 0;
%                 rec = rec + detail; %shorten rec correctly
%             end
        end
        
        function U = uncertainty(this, U_x, data)
        % Input:    U_x - uncertainty of the sensor, 
        %           data - one cycle
        % Output:   U - uncertainty matrix for wavelet coefficients

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
                
                % Vector with size of different levels
                l = this.len(2:end);
                l(end+1) = this.originLen;

                % Maximum number of levels
                max_level = size(this.len,2)-1;

                % Filter g low pass (for approximation vector) 
                % Filter h high pass (for detail vector)
                [g_filter, h_filter] = wfilters('db2');

                % Only one level
                A{1,1} = this.mat_filter(g_filter,l(end-1),l(end));
                C{1,1} = this.mat_filter(h_filter,l(end-1),l(end));

                % For 2 or more level
                for i = 2 : max_level
                    A{1,i} = this.mat_filter(g_filter,l(end-i),l(end-i+1))*A{1,i-1};
                    C{1,i} = this.mat_filter(h_filter,l(end-i),l(end-i+1))*A{1,i-1};
                end

                % Combine all C in one matrix
                % If only one level
                C_ges = C{1,1};
                % If more than one level
                for i = 2 : max_level
                    C_ges = vertcat(C{1,i},C_ges);
                end

                % For A, only the highest level is of interest
                A_ges = A{1,max_level};

                % Blockwise calculation of the uncertainty matrix
                U11 = A_ges * U_x * A_ges';
                U12 = A_ges * U_x * C_ges';
                U22 = C_ges * U_x * C_ges';

                % Uncertainty matrix (approximation / detail coefficients)
                U_tmp = [U11 U12; U12' U22];

                % Uncertainty values for the features (as row vector)
                U_tmp = sqrt(abs(diag(U_tmp)))';

                % Index of relevant features
                index = this.ind;

                % Uncertainty for relevant features
                U = U_tmp(index);
                U_out = [U_out; U];
            end 
            U = U_out;
        end
        
        function [mat] = mat_filter(this,filter,row,column)
        % Creates the matix representation for a filter

            % Initalize matrix
            mat = zeros(row,column);

            % Row 1 always consists of the same values
            mat(1,1) = sum(filter(2:4));
            mat(1,2) = filter(1);

            % Row 2 to Column fix((column-4)/2)+2 
            for i = 2 : fix((column-4)/2)+2
                mat(i,2*i-3:2*i) = flip(filter);
            end

            % 2 cases
            if mod(column,2)
                % odd number of columns
                % second last row
                mat(row-1,column-2) = filter(4);
                mat(row-1,column-1) = filter(3);
                mat(row-1,column) = sum(filter(1:2));
                % last row
                mat(row,column) = sum(filter(1:4));
            else
                % even number of columns, only the last row has to be
                % calculated
                mat(row,column-1) = filter(4);
                mat(row,column) = sum(filter(1:3));
            end
        end
    end
    
    methods (Access = protected)
        
        function finishTraining(this)
            mean = this.m ./ this.n;
            [mean, idx] = sort(mean, 'descend');
            this.idx = idx;
            i = false(size(mean));
            if isempty(this.numFeat) && isempty(this.heuristic)
                nFeat = floor(size(mean, 2)/10);
            elseif isempty(this.heuristic)
                nFeat = this.numFeat;
            elseif strcmp(this.heuristic,'elbow')
                nFeat = FeatureExtractorInterface.elbowPos(mean);
            end
            this.numFeat = nFeat;
            i(idx(1:nFeat)) = true;
            this.ind = i;
            this.trainingFinished = true;
        end
    end
    
    methods (Static)
        function wl = wMaxLv(len)
            if len <= 5
                wl = 0;
            else
                wl = 1;
                cur = 6;
                sum = 11;
                while sum < len
                    cur = cur * 2;
                    sum = sum + cur;
                    wl = wl + 1;
                end
            end
        end
    end
end

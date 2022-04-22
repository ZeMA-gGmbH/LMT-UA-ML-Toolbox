classdef RFESVMUnc < FSSuperClassUncertainty & Uncertainty
    %RFESVMEXTRACTOR Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
        function this = RFESVMUnc(numFeat)
            if nargin > 0
                if exist('numFeat', 'var') && ~isempty(numFeat)
                    this.nFeat = numFeat;
                end
            end
        end
        
        function train(this, X, Y, U)
            X = zscore(X);
            subsInd = true(1, size(X,2));
            nSelected = size(X,2);
            
            numToSel = 1;
            rank = inf(1,size(X,2));

            %delete features that have nan values
            while any(any(isnan(X(:,subsInd)))) && (nSelected > numToSel)
                ind = find(subsInd);
                ex = find(any(isnan(X(:,subsInd)), 1));
                ex = ex(1);
                subsInd(ind(ex)) = false;
                nSelected = nSelected - 1;
                if nargout == 2
                    rank(ind(ex)) = nSelected;
                end
            end
            
            while nSelected > numToSel
                % Ref: [1] J. Bi and T. Zhang, “Support Vector Classification with Input Data Uncertainty,” 
                % in Advances in Neural Information Processing Systems, 2004, vol. 17, [Online]. 
                % Available: https://proceedings.neurips.cc/paper/2004/file/22b1f2e0983160db6f7bb9f62f4dbb39-Paper.pdf.
                % Algorithm 1
                % 1. 
                t = templateSVM('KernelFunction', 'linear','IterationLimit',20,'SaveSupportVectors', true);
                mdl = fitcecoc(X(:,subsInd)+U(:,subsInd), Y, 'Coding', 'onevsone', 'Learners', t, 'Options', statset('UseParallel', true));
                % 2. beta bias
                w = zeros(length(mdl.BinaryLearners{1}.Beta), length(mdl.BinaryLearners));
                b = zeros(1,length(mdl.BinaryLearners{1}.Bias));
                for i = 1:length(mdl.BinaryLearners)
                    w(:,i) = mdl.BinaryLearners{i}.Beta;
                    b(1,i) = mdl.BinaryLearners{i}.Bias;
                end

                % 2. solve eq. 6 
                delta = zeros(1,size(U,2));
                for i = 1 : size(U,2)
                    delta(1,i) = norm(U(:,i));
                end
                
                codeMat = abs(mdl.CodingMatrix);
                targ = unique(Y);
                delta_x_opt = cell(1,length(mdl.BinaryLearners));
                for i = 1 : length(mdl.BinaryLearners)
                    tmp = targ(logical(codeMat(:,i)));
                    target_used = Y == tmp(1) | Y == tmp(2);
                    x = X(target_used,:);
                    y = Y(target_used);
                    delta_x_opt{1,i} = y*(delta(i)*w(i)/norm(w(i)));
                end
                
                weights = sum(abs(w),2);
                
                %eliminate worst feature
                ind = find(subsInd);
                [~, ex] = min(weights);
                subsInd(ind(ex)) = false;
                nSelected = nSelected - 1;
                rank(ind(ex)) = nSelected;
            end
            [~, this.rank] = sort(-rank, 'descend');
            this.nFeat = min(this.nFeat, size(X,2));
        end
    
        function infoCell = info(this)
            infoCell = cell(this.nFeat,2);
            for i = 1:this.nFeat
                infoCell{i,1} = [1];
                infoCell{i,2} = [this.rank(i)];
            end
        end
    
        function U = uncertainty(this, U_x, data)
            sorted = this.rank;
            U = U_x(:,sorted(1:this.nFeat));         
        end
    end
end


classdef LDA < SupervisedTrainable & Appliable & Uncertainty
    %LDACLASSIFIER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        projLDA = [];
        leftOutFeat = [];
        meanTrain = [];
    end
    
    methods
        function this = LDA(varargin)
        end
        
        function train(this, X, Y)
            %remove constant Features to prevent nans and Infs in
            %covariance matrices
            s = std(X, [], 1);
            this.leftOutFeat = s==0;
            X = X(:, ~this.leftOutFeat);
            
            groups = unique(Y);
            dim = length(groups) - 1;
            
            %X = zscore(X);
            xm = mean(X);
            this.meanTrain = xm;
            X = X - xm;
            
            withinSSCP = zeros(size(X,2));
            covarianz = cell(size(groups));
            gm = cell(size(groups));
            for g = 1:length(groups)
                if iscell(groups) 
                    ind = strcmp(Y, groups(g));
                else
                    ind = Y == groups(g);
                end
                gm{g} = mean(X(ind,:));
                withinSSCP = withinSSCP + (X(ind,:)-gm{g})' * (X(ind,:)-gm{g});
                covarianz{g} = cov(X(ind,:));
            end
            betweenSSCP = X' * X - withinSSCP;
            
            try
            warning('off')
            [proj, ~] = eig(withinSSCP\betweenSSCP);
            warning('on')
            proj = proj(:, 1:min(size(X,2),dim));
            scale = sqrt(diag(proj' * withinSSCP * proj) ./ (size(X,1)-length(groups)));
            proj = bsxfun(@rdivide, proj, scale');
            this.projLDA = proj;
            catch ME
                disp(getReport(ME));
            end
        end
        
        function proj = apply(this, X)
            X = X(:, ~this.leftOutFeat);
            %CenterData with mean from Prev?!
            xm = this.meanTrain;
            X = X - xm; 
            if isempty(this.projLDA)
                error('Train before apply');
            else
                try
                    proj = X * this.projLDA;
                catch ME
                    disp(ME)
                end
            end
        end
        
        function U = uncertainty(this, U_x, data)
            % Uncertainty for the LDA 
            U = sqrt(abs(U_x.^2*(this.projLDA.^2))); 
        end
    end
end


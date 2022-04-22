classdef RELIEFFMatlab< FSSuperClass
    %RELIEFFSELECTOR Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        numNearestNeighbors = 3;
        classification = true;
    end
    
    methods
        function this = RELIEFFMatlab(numFeat)
            if nargin > 0
                if exist('numFeat', 'var') && ~isempty(numFeat)
                    this.nFeat = numFeat;
                end
            end
        end
        
        function train(this, X, Y)
            X = zscore(X);
            if this.classification == true
                Y = categorical(Y);
            end
            [ind,~]= relieff(X,Y,this.numNearestNeighbors);
            
            this.rank = ind;
            this.nFeat = min(this.nFeat, size(X,2));
        end
        
        function infoCell = info(this)
            infoCell = cell(this.nFeat,2);
            for i = 1:this.nFeat
                infoCell{i,1} = [1];
                infoCell{i,2} = [this.rank(i)];
            end
        end
    end
end


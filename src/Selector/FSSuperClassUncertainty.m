classdef FSSuperClassUncertainty < Appliable & Ranking & SupervisedUncertaintyTrainable & Uncertainty
    
    properties (Access = public)
        nFeat = 500;
    end
    
    methods
        function feat = apply(this, X)
            feat = X(:,this.rank(1:this.nFeat));
        end
        
        function r = getRanking(this)
            r = this.rank;
        end
    end
end


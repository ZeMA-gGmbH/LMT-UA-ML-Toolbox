classdef Standardization < UnsupervisedTrainable & Appliable
    
    properties
        mu = [];
        sigma = [];
    end
    
    methods
        function this = Standardization()
        end
        
        function train(this, data)
            [~, this.mu, this.sigma] = zscore(data);
        end
        
        function z = apply(this,data)
            z = (data-this.mu)./this.sigma;
        end
    end
end


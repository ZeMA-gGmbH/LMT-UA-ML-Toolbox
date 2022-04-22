classdef PLSRUncertainty < SupervisedTrainable & Appliable & Uncertainty
    %PLSR Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        numComp = 5;
        beta = [];
    end
    
    methods
        function this = PLSRUncertainty(numComp)
            if nargin > 0
                while iscell(numComp)
                    numComp = numComp{1,1};
                end
                this.numComp = numComp;
            end
        end
        
        function train(this, data, target)
            if(size(data,2)<this.numComp)
                %warning("less number of Components");
                this.numComp = size(data,2);
            end
            [~,~,~,~,this.beta] = plsregress(data,target,this.numComp);
        end
        
        function pred = apply(this, data)
            pred = [ones(size(data,1),1) data]*this.beta;
        end

        function U = uncertainty(this, U_x, data)
            % Uncertainty for the PLSR result
            U = sqrt(abs([zeros(size(data,1),1) U_x].^2*(this.beta.^2))); 
        end
    end
end


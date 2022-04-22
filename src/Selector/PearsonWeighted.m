classdef PearsonWeighted < FSSuperClassUncertainty & Uncertainty
    %PEARSONSELECTOR Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        rank_coeff = [];
    end
    
    methods
        function this = PearsonWeighted(numFeat)
           if nargin > 0
               this.nFeat = numFeat;
           end
        end
        
        function train(this, X, Y, U) 
            % Uncertainty = 0, then weigthed Pearson with Identity
            if all(U(:)==0)
                U = ones(size(X));
            end
            % weighted Pearson
            r_w = zeros(size(X,2),1);
            for i = 1 : size(X,2)
                m_X = mean(U(:,i).*X(:,i));
                m_Y = mean(U(:,i).*Y(:,1));
                s_X = sum(U(:,i).*(X(:,i)-m_X).^2);
                s_Y = sum(U(:,i).*(Y(:,1)-m_Y).^2);
                s_XY = (U(:,i).*X(:,i)-m_X)'*(Y(:,1)-m_Y);
                r_w(i,1) = abs(s_XY/sqrt(s_X*s_Y));
            end
            r_w(isnan(r_w)) = 0;
            this.rank_coeff = r_w;
            [~, this.rank] = sort(r_w, 'descend');
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


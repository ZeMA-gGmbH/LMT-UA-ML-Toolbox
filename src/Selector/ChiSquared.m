classdef ChiSquared < FSSuperClassUncertainty
    % ChiSquared takes uncertainty for the features into account
    % ChiSquared score can be used to select the features with the
    % highest values for the test chi-squared statistic from X, which must
    % contain only non-negative features such as booleans or frequencies.
    % Recall that the chi-square test measures dependence between stochastic
    % variables, so using this function "weeds out" the features that are the
    % most likely to be independent of class and therefore irrelevant for
    % classification.
    
    properties
    end
    
    methods
        function this = ChiSquared(numFeat)
           if nargin > 0
               this.nFeat = numFeat;
           end
        end
        
        function train(this, X, Y, U)
            % Test
            load fisheriris
            tmp = Inf(size(species,1),1);
            for i = 1 : size(species,1)
                if strcmp(species{i,1},"setosa")
                    tmp(i) = 0;
                elseif strcmp(species{i,1},"versicolor")
                    tmp(i) = 1;
                else
                    tmp(i) = 2;
                end
            end
            X = meas;
            y = tmp;
            
            elements = unique(y);
            for i = 1 : size(elements,1)
                for j = 1 : size(y,1)
                    if y(j,1) == elements(i)
                        Y(j,i) = 1;
                    end
                end
            end
            observed = Y'*X;
            feature_count = sum(X);
            class_prob = mean(Y);
            expected = class_prob'.*feature_count;
            k = length(observed);
            chi2 = (((observed - expected)./U).^2)./expected;
            chi2 = sum(chi2,1);
            
            % a higher value of chi2 statistic means two categorical variables
            % are dependent and therefore more useful for classification
%             tmp = ((X-Y)./U_ext).^2; %nochmal über Gewicht nachdenken
%             chi2 = sum(tmp)';
            chi2(isnan(chi2)) = 0;
            [~, this.rank] = sort(chi2, 'descend');
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


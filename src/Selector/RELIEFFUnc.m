classdef RELIEFFUnc < FSSuperClassUncertainty & Uncertainty
    %RELIEFFSELECTOR Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        numNearestNeighbors = 3;
%         classification = true;
    end
    
    methods
        function this = RELIEFFUnc(numFeat)
            if nargin > 0
                if exist('numFeat', 'var') && ~isempty(numFeat)
                    this.nFeat = numFeat;
                end
            end
        end
        
        function train(this, X, Y, U)
%             X = zscore(X);
% 
%             if this.classification == true
%                 Y = categorical(Y);
%             end
%             [~,weight]= relieff(X,Y,this.numNearestNeighbors);
%             unc = sum(U,1);
%             weight_unc = weight./unc; 
%             [~,ind] = sort(weight_unc,'descend');
%             this.rank = ind;

            X = zscore(X);
            
            %Check, if there are at least four samples per group
            %(each one has three nearest neightbours)
            groups = unique(Y);
            numPerGroup = zeros(length(groups),1);
            for i = 1:length(groups)
                try
                    numPerGroup(i) = sum(Y == groups(i));
                catch
                    numPerGroup(i) = sum(strcmp(Y,groups{i}));
                end
            end
            n = min(numPerGroup);
            nNN = 3;
            if n <= 1
                error('empty group in reliefFUni');
            elseif n <= 3
                nNN = n - 1;
            end

            rank = zeros(1, size(X,2));

            for g = 1:length(groups)
                dist_m_kNN = [];
                dist_h_kNN = [];
                index_m_kNN =[];
                index_h_kNN = [];
                % Misses and Hits
                miss = X(Y ~= groups(g),:);
                hit = X(Y == groups(g),:);
                U_miss = U(Y ~= groups(g),:);
                U_hit = U(Y == groups(g),:);
                
                % Nearst Miss and Hits
                for i = 1 : size(hit,1)
                    dist_m = [];
                    dist_h = [];
                    index_m =[];
                    index_h = [];
                    % Manhatten distance for misses
                    [dist_m,index_m]=sort(U_miss.*sum(abs(miss-hit(i,:)),2));
                    dist_m_kNN(i,:) = dist_m(1:this.numNearestNeighbors,1)';
                    index_m_kNN(i,:) = index_m(1:this.numNearestNeighbors,1)';
                
                    % Manhatten distance for hits
                    [dist_h,index_h]=sort(U_hit.*sum(abs(hit-hit(i,:)),2));
                    dist_h_kNN(i,:) = dist_h(2:this.numNearestNeighbors+1,1)';
                    index_h_kNN(i,:) = index_h(2:this.numNearestNeighbors+1,1)';
                end

                for i = 1:this.numNearestNeighbors
                    rank = rank + sum(abs(X(Y == groups(g),:) - X(index_m_kNN(:,i),:)), 1) - sum(abs(X(Y == groups(g),:) - X(index_h_kNN(:,i),:)), 1); 
                end
            end

            [~, this.rank] = sort(rank, 'descend');
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


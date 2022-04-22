classdef LDAMahalClassifier < CLSuperClass & Uncertainty
    %LDACLASSIFIER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        data_plot = {}
        target_plot = {}
        projLDA = [];
        gm = [];
        icovar = [];
        target = [];
        leftOutFeat = [];
        meanTrain = [];
        max_pred = [];
        min_pred = [];
    end
    
    methods
        function this = Classifier(varargin)
            if narargin == 0
                this.data = [];
                this.target = [];
                this.projLDA = [];
            elseif narargin == 2
                this.data = varargin{1};
                this.target = varargin{2};
                this.projLDA = [];
            elseif narargin == 3
                this.data = varargin{1};
                this.target = varargin{2};
                this.projLDA = varargin{3};
            else
                error('Wrong number of arguments');
            end
        end
        
        function train(this, X, Y)
            tempData = cell(1,1);
            tempTarget = cell(1,1);
            tempData{1,1} = X;
            tempTarget{1,1} = Y;
            this.data_plot = tempData;
            this.target_plot = tempTarget;
            
            this.target = Y;
            
            %remove constant Features to prevent nans and Infs in
            %covariance matrices
            s = std(X, [], 1);
            this.leftOutFeat = s~=0;
            X = X(:, this.leftOutFeat);
            
            groups = unique(Y);
            dim = length(groups) - 1;
            
            %X = zscore(X);
            xm = mean(X);
            this.meanTrain = xm;
            X = X - xm(ones(size(X,1),1),:); 
            
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
            this.gm = gm;
            betweenSSCP = X' * X - withinSSCP;
            
            try
            warning('off')
            [proj, ~] = eig(withinSSCP\betweenSSCP);
            warning('on')
            proj = proj(:, 1:min(size(X,2),dim));
            scale = sqrt(diag(proj' * withinSSCP * proj) ./ (size(X,1)-length(groups)));
            proj = bsxfun(@rdivide, proj, scale');
            this.icovar = cellfun(@(a){inv(proj' * (a * proj))}, covarianz);
            this.projLDA = proj;
            catch ME
                disp(getReport(ME));
            end
        end
        
        function pred = apply(this, X)
            tempData = cell(size(this.data_plot,1)+1,1);
            tempTarget = cell(size(this.target_plot,1)+1,1);
            for i = 1:size(this.data_plot,1)
                tempData{i,1} = this.data_plot{i,1};
                tempTarget{i,1} = this.target_plot{i,1};
            end
            tempData{end,1} = X;
            this.data_plot = tempData;
            
            X = X(:, this.leftOutFeat);
            %CenterData with mean from Prev?!
            xm = this.meanTrain;
            X = X - xm(ones(size(X,1),1),:); 
            if isempty(this.projLDA)
                pred = ones(size(X,1),1) * mode(this.target);
            else
                try
                    groups = unique(this.target);
                    testDataProj = X * this.projLDA;
                    mahalDist = Inf(size(testDataProj,1), length(groups));
                    for g = 1:length(groups)
                        m = this.gm{g}*this.projLDA;
                        mahalDist(:,g) = sum(((testDataProj-m)*this.icovar{g}) .* (testDataProj-m), 2);
                    end
                    [~, predInd] = min(mahalDist, [], 2);
                    if ischar(groups)
                        pred = groups{predInd};
                    else
                        pred = groups(predInd);
                    end
                catch ME
                    disp(ME)
                    %try to predict into majority class
                    try
                        groups = unique(this.target);
                        counts = zeros(size(groups));
                        for g = 1:length(groups)
                            if iscell(groups)
                                counts(g) = sum(strcmp(this.target, groups{g}));
                            else
                                counts(g) = sum(this.target == groups(g));
                            end
                        end
                        [~, pred] = max(counts);
                        pred = repmat(pred, size(X,1), 1);
                    catch
                        pred(:) = nan;
                    end
                end
            end
            tempTarget{end,1} = pred;
            this.target_plot = tempTarget;
        end
        
        function showLDA(this, varargin)
            plot3 = false;
            if nargin > 1
                plot3 = varargin{1};
            end
            figure;
            if isempty(this.projLDA)
                title('LDA not trained or training error', 'FontSize', 16);
                return;
            end
            trans = this.data*this.projLDA(1:end-1,:)...
                        + ones(size(this.data,1), 1)*this.projLDA(end,:);
            groups = unique(this.target, 'stable');
            hold on
            for i = 1:length(groups)
                if iscell(groups)
                    ind = strcmp(groups{i}, this.target);
                else
                    ind = groups(i) == this.target;
                end
                if plot3 && size(trans,2) > 2
                    scatter3(trans(ind,1), trans(ind,2), trans(ind,3), 'filled', 'LineWidth', 2);
                elseif size(trans,2) > 1
                    scatter(trans(ind,1), trans(ind,2), 'filled', 'LineWidth', 2);
                else
                    histogram(trans(ind));
                end
            end
            if iscell(groups)
                legend(groups, 'Location', 'best');
            else
                legend(arrayfun(@num2str, groups, 'UniformOutput', false), 'Location', 'best');
            end
            set(gcf, 'PaperPositionMode', 'auto');
            xlabel('first discriminant function', 'FontSize', 16);
            ylabel('second discriminant function', 'FontSize', 16);
            if plot3
                zlabel('third discriminant function', 'FontSize', 16);
            end
        end
        
        function show(this)
            figure()
            numD = size(this.data_plot,1);
            first = true;

            for i = 1:size(this.data_plot,1)

                proj = (this.data_plot{i,1}-this.meanTrain(ones(size(this.data_plot{i,1},1),1),:))*this.projLDA;
                if first == true
                    scat = scatter(proj(:,1),proj(:,2),75,this.target_plot{i,1},'*');
                    first = false;
                    
                elseif i == 2 %Use for simple trainig stack because the
                %plot becomes more clear
                    
                else
                    scat = scatter(proj(:,1),proj(:,2),30,this.target_plot{i,1},'d','filled','MarkerEdgeColor',[i/numD 0 0],'LineWidth',0.5);
                end
                hold on
            end
            legend();
            colorbar();
            xlabel("DF1");
            ylabel("DF2");
            box on
            title("LDA (* traindata / diamonds testdata)")  
            
        end
        
        function showAxisData(this)
            figure()
            box on
            numD = size(this.data_plot,1);
            first = true;

            for i = 1:size(this.data_plot,1)

                proj = (this.data_plot{i,1}-this.meanTrain(ones(size(this.data_plot{i,1},1),1),:))*this.projLDA;
                if first == true
                    scat = scatter(proj(:,1),proj(:,2),200,this.target_plot{i,1},'.');
                    first = false;
                    
                elseif i == 2 %Use for simple trainig stack because the
                %plot becomes more clear
                
                else
                    %scat = scatter(proj(:,1),proj(:,2),30,this.target_plot{i,1},'o','filled','MarkerEdgeColor',[i/numD 0 0],'LineWidth',0.5);
                    scat = scatter(proj(:,1),proj(:,2),30,this.target_plot{i,1},'o');
                end
                hold on
            end
            box on
            legend off;
            hcb = colorbar;
            hcb.Title.String = 'Wear [%]';
            xlabel('DF 1', 'FontSize', 16)
            ylabel('DF 2', 'FontSize', 16)
            legend('Training data','Test data');
            title('LDA', 'FontSize', 20)
            
        end        

        function showUncertainty(this,U,data) 
            figure()
            box on
            hold on
            first = true;

            for i = 1:size(this.data_plot,1)

                if first == true
                    proj = (this.data_plot{i,1}-this.meanTrain(ones(size(this.data_plot{i,1},1),1),:))*this.projLDA;
                    scat = scatter(proj(:,1),proj(:,2),200,this.target_plot{i,1},'.');
                    first = false;
                    
                elseif i == 2 || i== 3 %Use for simple trainig stack because the
                %plot becomes more clear
                
                else
                    proj = (this.data_plot{end,1}-this.meanTrain(ones(size(this.data_plot{end,1},1),1),:))*this.projLDA;
                    yneg = U(:,2);
                    ypos = yneg;
                    xneg = U(:,1);
                    xpos = xneg;
                    % Plot every fifth value with error bar
                    a = 5;
                    scat = scatter(proj(:,1),proj(:,2),30,this.target_plot{i,1},'o','LineWidth',0.5);
                    errorbar(proj(1:a:end,1),proj(1:a:end,2),...
                        yneg(1:a:end),ypos(1:a:end),...
                        xneg(1:a:end),xpos(1:a:end),'o','color', 'black','LineWidth',1)
                    scat = scatter(proj(1:a:end,1),proj(1:a:end,2),30,this.target_plot{i,1}(1:a:end,1),'o','LineWidth',1);
                    box on
                end
                hold on
            end
            box on
            hcb = colorbar;
            hcb.Title.String = '';
            xlabel('DF 1', 'FontSize', 16)
            ylabel('DF 2', 'FontSize', 16)
            legend('Training data','Test data');
            
        end    
 
        function showUncertaintyAxisData(this,U,data) 
            figure()
            box on
            hold on
            first = true;

            for i = 1:size(this.data_plot,1)

                if first == true
                    proj = (this.data_plot{i,1}-this.meanTrain(ones(size(this.data_plot{i,1},1),1),:))*this.projLDA;
                    scat = scatter(proj(:,1),proj(:,2),200,this.target_plot{i,1},'.');
                    first = false;
                    
                elseif i == 2 || i== 3 %Use for simple trainig stack because the
                %plot becomes more clear
                
                else
                    proj = (this.data_plot{end,1}-this.meanTrain(ones(size(this.data_plot{end,1},1),1),:))*this.projLDA;
                    yneg = U(:,2);
                    ypos = yneg;
                    xneg = U(:,1);
                    xpos = xneg;
                    % Plot every fiftyth value with error bar
                    a = 150;
                    % a = 5;
                    scat = scatter(proj(:,1),proj(:,2),30,this.target_plot{i,1},'o','LineWidth',0.5);
                    errorbar(proj(1:a:end,1),proj(1:a:end,2),...
                        yneg(1:a:end),ypos(1:a:end),...
                        xneg(1:a:end),xpos(1:a:end),'o','color', 'black','LineWidth',1)
                    scat = scatter(proj(1:a:end,1),proj(1:a:end,2),30,this.target_plot{i,1}(1:a:end,1),'o','LineWidth',1);
                    box on
                end
                hold on
            end
            box on
            hcb = colorbar;
            hcb.Title.String = '';
            xlabel('DF 1', 'FontSize', 16)
            ylabel('DF 2', 'FontSize', 16)
            legend('Training data','Test data','Test data w/ uncertainty','Location','best');          
        end    
        
        function U = uncertainty(this, U_x, data)
            % Uncertainty for the LDA 
            U = sqrt(abs(U_x.^2*(this.projLDA.^2))); 
            
            data = data - this.meanTrain(ones(size(data,1),1),:); 
            if isempty(this.projLDA)
                pred = ones(size(data,1),1) * mode(this.target);
            else
                try
                    % Original point with no uncertainty
                    groups = unique(this.target);
                    testDataProj = data * this.projLDA;
                    
                    mahalDist = Inf(size(testDataProj,1), length(groups));
                    for g = 1:length(groups)
                        m = this.gm{g}*this.projLDA;
                        mahalDist(:,g) = sum(((testDataProj-m)*this.icovar{g}) .* (testDataProj-m), 2);
                    end
                    [~, predInd] = min(mahalDist, [], 2);
                    if ischar(groups)
                        pred = groups{predInd};
                    else
                        pred = groups(predInd);
                    end
                    
                    % Points at the edges, calculated with uncertainty,
                    % number of linear discrimininats is limited to 20
                    n_discriminants = min(20, size(testDataProj,2));
                    mat_combis = LDAMahalClassifier.U_combis(ff2n(n_discriminants),U);

                    % 2^dimension possibile edge points
                    combis = 2^size(testDataProj,2);
                    
                    % calculate Mahalanobis distance for each combination
                    mahalDist = cell(size(mat_combis,1), size(mat_combis,2));
                    for i = 1 : combis
                        tmp = testDataProj + mat_combis{i,1};
                        for g = 1:length(groups)
                            m = this.gm{g}*this.projLDA;
                            mahalDist{i,1}(:,g) = sum(((tmp-m)*this.icovar{g}) .* (tmp-m), 2);
                        end
                        [~, predInd(:,i)] = min(mahalDist{i,1}, [], 2);                       
                    end
                    
                    % Max/Min of all goups
                    max_predInd = max(predInd, [], 2);
                    min_predInd = min(predInd, [], 2); 
                    
                    % Rename with the original group names
                    if ischar(groups)
                        max_pred = groups{max_predInd};
                        min_pred = groups{min_predInd};
                    else
                        max_pred = groups(max_predInd);
                        min_pred = groups(min_predInd);
                    end 
                    
                    % Save in property
                    this.max_pred = max_pred;
                    this.min_pred = min_pred;
                  
                catch ME
                    disp(ME)
                    %try to predict into majority class
                    try
                        groups = unique(this.target);
                        counts = zeros(size(groups));
                        for g = 1:length(groups)
                            if iscell(groups)
                                counts(g) = sum(strcmp(this.target, groups{g}));
                            else
                                counts(g) = sum(this.target == groups(g));
                            end
                        end
                        [~, pred] = max(counts);
                        pred = repmat(pred, size(data,1), 1);
                    catch
                        pred(:) = nan;
                    end
                end
            end
            
        end
    end
    
    methods(Static)    
        % Function for calculating all possible combinations to get the
        % edges of the error bars
        function possible_mat = U_combis(combis,U) 
            % Transfer 0 and 1 to + and -
            combis(combis==1) = -1;
            combis(combis==0) = 1;
            
            % Columnwise consideration of the signs
            possible_mat = cell(size(combis,1),1);
            for i = 1 : size(combis,1)
                for j = 1 : size(combis,2)
                    possible_mat{i,1}(:,j) = combis(i,j).*U(:,j);
                end
            end
        end
    end
    
end


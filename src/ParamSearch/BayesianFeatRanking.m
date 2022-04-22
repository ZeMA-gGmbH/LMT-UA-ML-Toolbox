classdef BayesianFeatRanking < SupervisedTrainable & Appliable
    %NUMFEATRANKING Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (Constant)
        maxFeat = 500;
    end
    
    properties
        rankingAlg = @Pearson;
        ranking = [];
        loss = [];
        lossAlg = @ClassificationError.loss;
        predictionStack = {@LDAMahalClassifier};
        predArg = {{}};
        nFeat = 0;
        bayesoptArgs = {'UseParallel',true};
        bayesianOpt = {};
    end
    
    methods
        
        function infoCell = info(this)
            infoCell = cell(this.nFeat,2);
            for i = 1:this.nFeat
                infoCell{i,1} = [1];
                infoCell{i,2} = [this.ranking(i)];
            end
        end
        
        function this = BayesianFeatRanking(rankingAlg, predictionStack, predArg, lossAlg, varargin)
            if nargin > 0
                if exist('rankingAlg', 'var') && ~isempty(rankingAlg)
                    this.rankingAlg = rankingAlg;
                end
                if exist('predictionStack', 'var') && ~isempty(predictionStack)
                    this.predictionStack = predictionStack;
                end
                if exist('predArg', 'var') && ~isempty(predArg)
                    this.predArg = predArg;
                end
                if exist('lossAlg', 'var') && ~isempty(lossAlg)
                    this.lossAlg = lossAlg;
                end
                if exist('varargin', 'var') && ~isempty(varargin)
                    this.bayesoptArgs = varargin;
                end
            end
        end
        
        function train(this, data, target)
            %get ranking
            rAlg = feval(this.rankingAlg);
            rAlg.train(data,target);
            this.ranking = rAlg.getRanking();
            
            %compute loss for each feature number
            cv = cvpartition(target, 'KFold', 10);
            
            
            predStack = this.predictionStack;
            pArg = this.predArg;
            r = this.ranking;

            evaluator = CrossValidator(predStack, pArg);
            Num_Feat = optimizableVariable('Num_Feat',[1, min(this.maxFeat, size(data,2))],'Type','integer');
            fun = @(x) feval(this.lossAlg, evaluator.crossValidate(data(:, r(1:x.Num_Feat)), target, cv), target);
            this.bayesianOpt = bayesopt(fun, Num_Feat, this.bayesoptArgs{:});
            this.loss = this.bayesianOpt.EstimatedObjectiveMinimumTrace;
            %get optimal number of features
            this.nFeat = this.bayesianOpt.XAtMinEstimatedObjective.Num_Feat;
        end
        
        function pred = apply(this, data)
            pred = data(:, this.ranking(1:this.nFeat));
        end
    end
end
classdef BayesianOptimizer < SupervisedTrainable & Appliable
    % Summary of this class goes here
    % Detailed explanation goes here
    properties (Constant)
        maxFeat = 500;
    end
    
    properties
        extractionAlgs = {@ALAExtractor, @BDWExtractor, @BFCExtractor, @PCAExtractor, @StatisticalMoments};
        rankingAlgs = {@Pearson, @RELIEFF, @RFESVM};
        classifiers = {@LDAMahalClassifier, @SVMClassifier};
        lossAlg = @ClassificationError.loss;
        optVars = [];
        optResults = [];
        trainingStack = [];
        searchhistory = [];
        already_tested_extractor = [];
        isFeatureData=false;
        bayesoptArgs = {'UseParallel',true};
    end
    
    methods
        function this = BayesianOptimizer(extrAlgs, selAlgs, class, varargin)
            if nargin > 0
                if exist('extrAlgs', 'var') && ~isempty(extrAlgs)
                    this.extractionAlgs = extrAlgs;
                end
                if exist('selAlgs', 'var') && ~isempty(selAlgs)
                    this.rankingAlgs = selAlgs;
                end
                if exist('class', 'var') && ~isempty(class)
                    this.classifiers = class;
                end
                if exist('varargin', 'var') && ~isempty(varargin)
                    this.bayesoptArgs = varargin;
                end
            end
        end
        %% train function
        function train(this, data, target)
            %varargin: all possible Name, Value inputs for function bayesopt()
            if ~iscell(data)
                data = {data};
            end
            %define optimization function and optimization variable
            %x: all optimization variables
            %y: all variables that are restricted to one possible value
            %define optimizableVariables Number of Features
            for ii = 1:2:length(this.bayesoptArgs)
                if this.bayesoptArgs{ii}=="isFeatureData"
                    this.isFeatureData = this.bayesoptArgs{ii+1};
                    this.bayesoptArgs(ii:ii+1) = [];
                    break
                end
            end
            
            if isempty(this.optResults)
                %Start new optimiziing
                if ~this.isFeatureData
                    %optimizableVariables Extraction Algorithm for every Sensor
                    range = this.extractionAlgs;
                    for i=1:size(data, 2)
                        if length(range) == 1
                            y.(['Extr' num2str(i)]) = range;
                            this.extractionAlgs = cell2struct(range, char(range{1}));
                        else
                            [ExtrOut, this.extractionAlgs] = this.functionhandleOptimizableVariable(['Extr' num2str(i)], range, 'categorical');
                            x.(ExtrOut.Name) = ExtrOut;
                            this.optVars = [this.optVars ExtrOut];
                        end
                    end
                    Num_Feat_max = this.maxFeat;
                else
                    data = cell2mat(data);
                    Num_Feat_max = min(size(data,2), this.maxFeat);
                end
                
                x.Num_Feat = optimizableVariable('Num_Feat',[1, Num_Feat_max],'Type','integer');
                this.optVars = [this.optVars x.Num_Feat];
                y=[];
                for Algorithm = ["classifiers", "rankingAlgs"]
                    %if >1 Algorithms are given -> create optimizableVariable
                    if length(this.(Algorithm)) == 1
                        y.(Algorithm) = this.(Algorithm);
                        this.(Algorithm) = cell2struct(this.(Algorithm), char(this.(Algorithm){1}));
                    else
                        [x.(Algorithm), this.(Algorithm)] = this.functionhandleOptimizableVariable(Algorithm, this.(Algorithm), 'categorical');
                        this.optVars = [this.optVars x.(Algorithm)];
                    end
                end
                
                %optimization function
                fun = @(x) this.objectiveFun(x, y, data, target);
                this.optResults = bayesopt(fun, this.optVars, this.bayesoptArgs{:});
            else
                %resume training if trainingsobject is already existing
                this.optResults = resume(this.optResults, this.bayesoptArgs{:});
            end
            
            bestEstimation = bestPoint(this.optResults);
            ExtAlgs = [];
            for i = 1:sum(contains(bestEstimation.Properties.VariableNames, 'Extr'))
                 ExtAlgs = [ExtAlgs bestEstimation.(['Extr' num2str(i)])];
            end
            %get Prediction Stack from bayes optimization results and
            %restricted points
            if ~isempty(ExtAlgs)
                if length(fieldnames(this.extractionAlgs)) > 1
                    ExtractionAlgs = cellfun(@(f) this.extractionAlgs.(f),cellstr(ExtAlgs),'UniformOutput',false);
                    ExtractionArgs = cellfun(@(f) {}, cellstr(ExtAlgs), 'UniformOutput', false);
                else
                    ExtractionAlgs = struct2cell(this.extractionAlgs);
                    ExtractionAlgs = {ExtractionAlgs{1}};
                    ExtractionArgs = {{}};
                end
                predictionArg= {{ExtractionAlgs, ExtractionArgs}, {500}, {bestEstimation.Num_Feat}, {}};
                predictionStack = {@MultiSensorSingleExtractor, @Pearson};
            else
                predictionArg={{500}, {bestEstimation.Num_Feat}, {}};
                predictionStack = {@Pearson};
            end
            for Algs = ["rankingAlgs", "classifiers"]
                if length(fieldnames(this.(Algs))) > 1
                    predictionStack{length(predictionStack)+1} = this.(Algs).(char(bestEstimation.(Algs)));
                else
                    Alg = struct2cell(this.(Algs));
                    predictionStack{length(predictionStack)+1} = Alg{1};
                end
            end

            this.trainingStack = SimpleTrainingStack(predictionStack, predictionArg);
            this.trainingStack.train(data,target);
            this.searchhistory = this.optResults.XTrace;
        end
        %% optimizing function
        function classError = objectiveFun(this, x, y, data, target)
            %returns classification error of given prediction stack
            for field = ["rankingAlgs", "classifiers"]
                if sum(contains(fieldnames(x), field))
                    %field is optimizable 
                    z.(field) = x.(field);
                else
                    %field is fix
                    z.(field) = this.function_handle2cat(y.(field){1});
                end
            end
            if ~this.isFeatureData
                z.Extr = [];
                if sum(contains(fieldnames(x), 'Extr')) > 0
                    %Extractor is optimizable
                    for j = 1:sum(contains(fieldnames(x), 'Extr'))
                         z.Extr = [z.Extr x.(['Extr' num2str(j)])];
                    end
                else
                    %Extractor is fix
                    for j = 1:sum(contains(fieldnames(y), 'Extr'))
                         z.Extr = [z.Extr this.function_handle2cat(y.(['Extr' num2str(j)]){1})];
                    end
                end

                extAlgs = cellfun(@(f) this.extractionAlgs.(f),cellstr(z.Extr),'UniformOutput',false);
                extArgs = cellfun(@(f) {}, cellstr(z.Extr), 'UniformOutput', false);

                %extract features from data
                feat = [];
                for n = 1:length(z.Extr)
                    sensor = 'sensor'+string(n);
                    Extr_Alg = string(z.Extr(n));
                    if ~isfield(this.already_tested_extractor, sensor)
                        this.already_tested_extractor.(sensor) = [];
                    end
                    %if extracor already used for this sensor: get features
                    %if not: train extrator, save extractor, get features
                    if isfield(this.already_tested_extractor.(sensor), Extr_Alg)
                        feat = [feat this.already_tested_extractor.(sensor).(Extr_Alg).apply(data{n})];
                    else
                        this.already_tested_extractor.(sensor).(Extr_Alg) = SimpleTrainingStack(extAlgs(n), extArgs(n));
                        this.already_tested_extractor.(sensor).(Extr_Alg).train(data{n});
                        feat = [feat this.already_tested_extractor.(sensor).(Extr_Alg).apply(data{n})];
                    end
                end
                if x.Num_Feat > size(feat,2)
                    %return 100% Error when tried Number of Features is
                    %greater than actually extracted features
                    x.Num_Feat = size(feat,2);
%                       classError = 100 + log(x.Num_Feat-size(feat,2));
                end 
            else
                feat = data;
            end

            if x.Num_Feat > size(feat,2)
                %return 100% Error when tried Number of Features is
                %greater than actually extracted features
                x.Num_Feat = size(feat,2);
%                     classError = 100 + log(x.Num_Feat-size(feat,2));
            end
            predStack = {@Pearson, this.rankingAlgs.(char(z.rankingAlgs)), this.classifiers.(char(z.classifiers))};
            predArg= {{500}, {x.Num_Feat}, {}};
            cv = cvpartition(target, 'KFold', 10);
            evaluator = CrossValidator(predStack, predArg);
            cvPred = evaluator.crossValidate(feat, target, cv);
            classError = feval(this.lossAlg, cvPred, target);

        end

        function [FunctionHandleOptVar, FunctionHandleBib] = functionhandleOptimizableVariable(this, Name, range, Type)
            FunktionNames = cell(1, size(range, 2));
            for j = 1:size(range, 2)
                FunktionNames{j} = char(range{j});
            end
            FunctionHandleBib = cell2struct(range, FunktionNames,2);
            FunctionHandleOptVar = optimizableVariable(Name, FunktionNames, 'Type', Type);
        end

        function cat = function_handle2cat(this, functionhandle)
            cat = categorical(string(char(functionhandle)));
        end
        
        %% apply function
        function pred = apply(this, data)
            if this.isFeatureData
                data = cell2mat(data)
            end
            pred = this.trainingStack.apply(data);
        end
    end
end


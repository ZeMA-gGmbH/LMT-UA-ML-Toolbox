classdef DNNSuperClass < SupervisedTrainable & Appliable
    %DNNSUPERCLASS Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        options = trainingOptions('adam','maxepoch',1) ;
        lgraph;
        inputDimention = [1 1000];
        outputDimention = 2;
        downsampleFactor =1;
        downSamplePerSensor = [];
        trainingHistory;
        trainedNet; 
        classification=true;
        depth=3; 
        reshape1D = true;
        filter2D = false;
        numSensors = 1;
        sensorLens = [1000];
    end
    
 methods
        function this = setHyperparameters(this,varargin)
            propertyList =  properties(this);
             p = inputParser;
            for i = 1:length(propertyList)
                addParameter(p, propertyList{i}, this.(propertyList{i}));
            end

            parse(p, varargin{:});
            for i = 1:length(p.Parameters)
                if ~any(strcmp(p.UsingDefaults, p.Parameters(i) ))
                    this.(p.Parameters{i}) = p.Results.(p.Parameters{i});
                end
            end
         end

        function train(this, data, target)
            if this.reshape1D == true
                data = this.reshapeData(data);
            else
                data = this.reshapeData2D(data);
            end
            if this.classification == true
                target = categorical(target);
                this.outputDimention = length(unique(target));
            else
                this.outputDimention = 1;
            end
            this.buildGraph();
            [this.trainedNet, this.trainingHistory] = trainNetwork(data, target, this.lgraph, this.options);
        end
        
        function pred = apply(this, data)
            data = this.reshapeData(data);
            if this.classification ==true
                pred = classify(this.trainedNet, data);
            else
                pred = predict(this.trainedNet, data);
            end
        end
        
        function [lgraph, trainData4D, validationData4D] = prepareExperiment(this, trainTarget, trainData, validationData)
             if this.classification == true
                this.outputDimention = length(unique(trainTarget));
             else
                this.outputDimention = 1;
             end
             trainData4D = this.reshapeData(trainData);
             validationData4D = this.reshapeData(validationData);
             lgraph = this.buildGraph();
        end
        
        
        function dataMat4D = reshapeData(this, data)
            this.numSensors = length(data);
            if length(this.downSamplePerSensor)~=this.numSensors
                this.downSamplePerSensor = ones(this.numSensors,1)*this.downsampleFactor;
            end
            numberSamples = size(data{1},1);
            
            this.sensorLens = [];
            for i = 1:this.numSensors
                data{i} = downsample(data{i}',this.downSamplePerSensor(i))';
                this.sensorLens(i) = floor(size(data{i},2));
            end
            data = cell2mat(data);
            width = size(data,2);
            dataMat4D = zeros(1,width,1,numberSamples);
            for i = 1:numberSamples
                dataMat4D(1,:,1,i) = data(i,:); 
            end
            this.inputDimention = size(dataMat4D);
            this.inputDimention = this.inputDimention(1:2);
        end
        
         function dataMat4D = reshapeData2D(this, data)
            numberSamples = size(data{1},1);
            height = length(data);
            for h = 1:height
                 data{h} = downsample(data{h}',this.downsampleFactor)';
            end
            
            width = size(data{1},2);
            dataMat4D = zeros(height,width,1,numberSamples);
            for i = 1:numberSamples
                for h = 1:height
                    dataMat4D(h,:,1,i) = data{h}(i,:); 
                end
            end
            this.inputDimention = size(dataMat4D);
            this.inputDimention = this.inputDimention(1:2);
        end
    end
end


classdef LSTMNetwork<DNNSuperClass
    %MLPNETWORK Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        numNeurons = 100;
        dropOut=0.5;
        numLSTMLayers = 1;
        numHiddenUnits = 100;
    end
    
    methods
        function this = LSTMNetwork(depth)
            this.depth = depth;
            this.options = trainingOptions('adam','maxepoch',1);
        end
        
        function lgraph = buildGraph(this)
            this.lgraph = this.lstmFun();
            lgraph = this.lgraph;
        end
        
        function dataCell = reshapeData(this, data)
            dataMat4D= reshapeData@DNNSuperClass(this, data);
            dataCell = cell(size(dataMat4D,4),1);
            for i = 1:length(dataCell)
                dataCell{i} = dataMat4D(:,:,:,i);
            end
        end

        function lgraph = lstmFun(this)

            input_shape = [this.inputDimention 1];
            layers = [
                sequenceInputLayer(input_shape,'Normalization','zscore','NormalizationDimension','auto','name','InputLayer')
                flattenLayer('name','flatten')
            ];
            for i = 1:this.numLSTMLayers-1
                layers = [layers;bilstmLayer(floor(this.numHiddenUnits/i),'OutputMode','sequence','name',['lstm',int2str(i)])];
            end
             layers = [layers;bilstmLayer(floor(this.numHiddenUnits/this.numLSTMLayers),'OutputMode','last','name','lstmFinal')];
            for i = 1:this.depth
                layers = [layers;this.fullyConUnit(ceil(this.numNeurons/(2*i)),['Block',int2str(i)])];
                layers = [layers;reluLayer('Name', ['relu',int2str(i)])];
            end

            layers = [layers;dropoutLayer(this.dropOut,'name','DropOut1')];
            layers = [layers;this.fullyConUnit(this.outputDimention,'Final')];
            if this.classification == false
                layers = [layers;regressionLayer('Name','routput')];
            else
                layers = [layers;softmaxLayer('Name','softmax')];
                layers = [layers;classificationLayer('Name','classoutput')];
            end
            lgraph = layerGraph(layers);
        end

        function layers = fullyConUnit(~, numF,tag)

            layers = [
                fullyConnectedLayer(numF,'Name',[tag 'fc'])
                batchNormalizationLayer('Name',[tag,'BN1'])
                reluLayer('Name',[tag 'relu fc'])
            ];

        end
    end
end


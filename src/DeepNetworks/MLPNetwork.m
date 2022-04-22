classdef MLPNetwork<DNNSuperClass
    %MLPNETWORK Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        n_neurons = 100;
        dropOut=0.5;

    end
    
    methods
        function this = MLPNetwork(depth)
            this.depth = depth;
            this.options = trainingOptions('adam','maxepoch',1);
        end
        
        function lgraph = buildGraph(this)
            this.lgraph = this.fcFun();
            lgraph = this.lgraph;
        end

        function lgraph = fcFun(this)

            input_shape = [this.inputDimention 1];

            layers = [
                imageInputLayer(input_shape,'Normalization','zscore','NormalizationDimension','auto','name','InputLayer')
            ];

            for i = 1:this.depth
                layers = [layers;this.fullyConUnit(ceil(this.n_neurons/(2*i)),['Block',int2str(i)])];
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


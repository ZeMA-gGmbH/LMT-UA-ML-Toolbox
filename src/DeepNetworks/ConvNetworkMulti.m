classdef ConvNetworkMulti  < DNNSuperClass
    %CONVNETWORK Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        n_filters = 8;
        firstKernel_size = 16;
        firstStride = 3;
        kernel_size= 5;
        stride_size= 2;
        dropOut = 0.5;
        numNeurons = 100;
        numNeuronsVector = [100,100];
        laneDepth = [3,3];
        
    end
    
    methods
         function this = ConvNetworkMulti(depth)
            this.depth = depth;
            this.options = trainingOptions('adam','maxepoch',1);
        end
        
        function lgraph = buildGraph(this)
            if length(this.numNeuronsVector)~= this.numSensors
                 this.numNeuronsVector = ones(1,this.numSensors)*this.numNeurons;
            end
            if length(this.laneDepth)~= this.numSensors
                 this.laneDepth = ones(1,this.numSensors)*this.depth;
            end
            this.lgraph = this.convFun();
            lgraph = this.lgraph;
        end
       function lgraph = convFun(this)

            input_shape = [this.inputDimention 1];

            inputLayers = [
                imageInputLayer(input_shape,'Normalization','zscore','NormalizationDimension','auto','name','InputLayer')
                DevideLayer(this.numSensors, 'devideSensors', this.sensorLens)
            ];
            layers = cell(this.numSensors,1);
        for s = 1:this.numSensors
                  layers{s} =  this.convolutionalUnit(this.n_filters,['Sen', int2str(s),'First Block'],this.firstStride,this.firstKernel_size);
            for i = 1:this.laneDepth(s)-1
                layers{s} = [layers{s};this.convolutionalUnit(this.n_filters*i,['Sen', int2str(s),'Block',int2str(i+1)],this.stride_size,this.kernel_size)];
            end
            %TODO maybe add averagePooling2dLayer
            layers{s} = [layers{s};this.fullyConUnit(this.numNeuronsVector(s),int2str(s))];
        end
            outputLayers = [concatenationLayer(3, this.numSensors,'Name','ConCat')
                dropoutLayer(this.dropOut,'name','DropOut1')];
            outputLayers = [outputLayers;this.fullyConUnit(this.outputDimention,'Final')];
            if this.classification == false
                outputLayers = [outputLayers;regressionLayer('Name','RegOutput')];
            else
                outputLayers = [outputLayers;softmaxLayer('Name','Softmax')];
                outputLayers = [outputLayers;classificationLayer('Name','ClassOutput')];
            end
            lgraph = layerGraph(inputLayers);
            lgraph = addLayers(lgraph,outputLayers);
            for s = 1:this.numSensors
                lgraph = addLayers(lgraph,layers{s});
                lgraph = connectLayers(lgraph,[inputLayers(end).Name,'/out',int2str(s)], layers{s}(1).Name);
                lgraph = connectLayers(lgraph, layers{s}(end).Name, [outputLayers(1).Name,'/in',int2str(s)]);
            end
            
       end
        function layers = convolutionalUnit(this, numF,tag,stride,filter)
            
            if this.filter2D
                stride = [stride, stride];
                filter = [filter, filter];
            else
                stride = [1, stride];
                filter = [1, filter];
            end    
            layers = [
                convolution2dLayer(filter,numF,'Padding','same','Stride',stride,'Name',[tag,'Conv1'])
                batchNormalizationLayer('Name',[tag,'BN1'])
                reluLayer('Name',[tag,'Relu1'])
            ];

        end

        function layers = fullyConUnit(~, numF,tag)

            layers = [
                fullyConnectedLayer(numF,'Name',[ 'FC' tag])
                reluLayer('Name',[ 'Relu FC' tag])
            ];
        end
    end
end


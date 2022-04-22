classdef ConvNetwork  < DNNSuperClass
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
    end
    
    methods
         function this = ConvNetwork(depth)
            this.depth = depth;
            this.options = trainingOptions('adam','maxepoch',1);
        end
        
        function lgraph = buildGraph(this)
            this.lgraph = this.convFun();
            lgraph = this.lgraph;
        end
       function lgraph = convFun(this)

            input_shape = [this.inputDimention 1];

            layers = [
                imageInputLayer(input_shape,'Normalization','zscore','NormalizationDimension','auto','name','InputLayer')
                this.convolutionalUnit(this.n_filters,'First Block',this.firstStride,this.firstKernel_size)
            ];

            for i = 1:this.depth-1
                layers = [layers;this.convolutionalUnit(this.n_filters*i,['Block',int2str(i+1)],this.stride_size,this.kernel_size)];
            end
            %TODO maybe add averagePooling2dLayer
            layers = [layers;this.fullyConUnit(this.numNeurons,'1')];
            layers = [layers;dropoutLayer(this.dropOut,'name','DropOut1')];
            layers = [layers;this.fullyConUnit(this.outputDimention,'Final')];
            if this.classification == false
                layers = [layers;regressionLayer('Name','RegOutput')];
            else
                layers = [layers;softmaxLayer('Name','Softmax')];
                layers = [layers;classificationLayer('Name','ClassOutput')];
            end
            lgraph = layerGraph(layers);
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


classdef ResNetNetwork<DNNSuperClass
    %RESNETNETWORK Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        n_filters = 8;
        firstKernel_size = 16;
        firstStride= 3;
        kernel_size= 5;
        stride_size=2;
        dropOut = 0.5
        numNeurons = 100;
    end
    
    methods
        function this = ResNetNetwork(depth)
            this.depth = depth;
            this.options = trainingOptions('adam','maxepoch',1);
        end
        
         function lgraph = buildGraph(this)
            this.lgraph = this.resNetFun();
            lgraph = this.lgraph;
        end
        
       function lgraph = resNetFun(this)

            input_shape = [this.inputDimention 1];

            layers = [
                imageInputLayer(input_shape,'Normalization','zscore','NormalizationDimension','auto','name','InputLayer')
                this.convolutionalUnit(this.n_filters, 'First Block', this.firstStride, this.firstKernel_size)
            ];

            for i = 1:this.depth-1
                connections{i,1} = layers(end).Name;
                layers = [layers; this.convolutionalUnit(this.n_filters*i, ['Block',int2str(i)], this.stride_size, this.kernel_size)];
                layers = [layers;additionLayer(2,'Name',['Addition ',int2str(i)])];
                connections{i,2} = [layers(end).Name,'/in2'];
                layers = [layers;reluLayer('Name',['Addition ReLU',int2str(i)])];
            end

            %TODO maybe add averagePooling2dLayer
            layers = [layers; this.fullyConUnit(this.numNeurons,'First')];
            layers = [layers;dropoutLayer(this.dropOut,'name','DropOut1')];
            layers = [layers; this.fullyConUnit(this.outputDimention,'Final')];
            if this.classification == false
                layers = [layers;regressionLayer('Name','RegOut')];
            else
                layers = [layers;softmaxLayer('Name','Softmax')];
                layers = [layers;classificationLayer('Name','ClassOut')];
            end

            lgraph = layerGraph(layers);

            for i = 1: this.depth-1
                lgraph = addLayers(lgraph, this.skipUnit(this.n_filters*i,['Block',int2str(i)], this.stride_size, this.kernel_size));
                lgraph = connectLayers(lgraph,connections{i,1},['Block',int2str(i),'skipconv1']);
                lgraph = connectLayers(lgraph,['Block',int2str(i),'skipBN2'],connections{i,2});
            end
        end

        function layers = convolutionalUnit(~, numF,tag,stride,filterSize)

            layers = [
                convolution2dLayer([1,filterSize],numF,'Padding','same','Stride',[1,stride],'Name',[tag,'Conv1'])
                batchNormalizationLayer('Name',[tag,'BN1'])
                reluLayer('Name',[tag,'relu1'])
                convolution2dLayer([1,filterSize],numF,'Padding','same','Stride',[1,1],'Name',[tag,'Conv2'])
                batchNormalizationLayer('Name',[tag,'BN2'])
            ];

        end

        function layers = skipUnit(~, numF,tag,stride,filterSize)

            layers = [
                convolution2dLayer([1,filterSize],numF,'Padding','same','Stride',[1,stride],'Name',[tag,'skipconv1'])
                batchNormalizationLayer('Name',[tag,'skipBN1'])
                reluLayer('Name',[tag,'skiprelu1'])
                convolution2dLayer([1,filterSize],numF,'Padding','same','Stride',[1,1],'Name',[tag,'skipconv2'])
                batchNormalizationLayer('Name',[tag,'skipBN2'])
            ];

        end

        function layers = fullyConUnit(~, numF,tag)

            layers = [
                fullyConnectedLayer(numF,'Name',[tag 'fc'])
                reluLayer('Name',[tag 'relu fc'])
            ];

        end
    end
end


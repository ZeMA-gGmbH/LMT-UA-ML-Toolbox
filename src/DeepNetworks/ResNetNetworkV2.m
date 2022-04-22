classdef ResNetNetworkV2<DNNSuperClass
    %RESNETNETWORK Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        numFilters = 8;
        firstFilterSize = 16;
        firstStride= 3;
        filterSize= 5;
        stride2=2;
        stride3=2;
        unitType = "standard";
    end
    
    methods
         function this = ResNetNetworkV2(depth)
            this.depth = depth;
            this.options = trainingOptions('adam','maxepoch',1);
        end
         function lgraph = buildGraph(this)
            this.lgraph = this.resNetFun();
            lgraph = this.lgraph;
        end
        
       function lgraph = resNetFun(this)
         stride1 = 1;
         if  this.filter2D
            firstFilterSize = this.firstFilterSize;
            firstStride = this.firstStride;
            stride2 = this.stride2;
            stride3 = this.stride3;
            filterSize  = this.filterSize;
         else
            firstFilterSize = [1, this.firstFilterSize];
            firstStride = [1,this.firstStride];
            stride2 = [1,this.stride2];
            stride3 = [1,this.stride3];
            filterSize  = [1,this.filterSize];
         end
         
         numFilters = this.numFilters;
             
         input_shape = [this.inputDimention 1];
        outputDimention = 10;
        unitsPerStage = floor(this.depth/3);
        if this.unitType == "standard"
            convolutionalUnit = @this.standardConvolutionalUnit;
        elseif this.unitType == "bottleneck"
            convolutionalUnit = @this.bottleneckConvolutionalUnit;
        else
            error("Residual block type must be either ""standard"" or ""bottleneck"".")
        end
        %% Create Main Network Branch
        %
        % Input section. Add the input layer and the first convolutional layer.
        layers = [
            imageInputLayer(input_shape,'Name','input')
            convolution2dLayer(firstFilterSize,numFilters,'stride',firstStride,'Padding','same','Name','convInp')
            batchNormalizationLayer('Name','BNInp')
            reluLayer('Name','reluInp')];
        % Stage one. Activation size is X1-by-X1.
        for i = 1:unitsPerStage
            layers = [layers
                convolutionalUnit(filterSize,numFilters,stride1,['S1U' num2str(i) '_'])
                additionLayer(2,'Name',['add1' num2str(i)])
                reluLayer('Name',['relu1' num2str(i)])];
        end
        % Stage two. Activation size is X2-by-X2.
        for i = 1:unitsPerStage
            if i==1
                stride = stride2;
            else
                stride = 1;
            end
            layers = [layers
                convolutionalUnit(filterSize,2*numFilters,stride,['S2U' num2str(i) '_'])
                additionLayer(2,'Name',['add2' num2str(i)])
                reluLayer('Name',['relu2' num2str(i)])];
        end
        % Stage three. Activation size is X3-by-X3
        for i = 1:unitsPerStage
            if i==1
                stride = stride3;
            else
                stride = 1;
            end
            layers = [layers
                convolutionalUnit(filterSize,4*numFilters,stride,['S3U' num2str(i) '_'])
                additionLayer(2,'Name',['add3' num2str(i)])
                reluLayer('Name',['relu3' num2str(i)])];
        end
        % Output section.
        layers = [layers
            globalAveragePooling2dLayer('Name','globalPool') %averagePooling2dLayer(pooling,'Name','globalPool')
            fullyConnectedLayer(this.outputDimention,'Name','fcFinal')];
            if this.classification == false
                layers = [layers;regressionLayer('Name','RegOut')];
            else
                layers = [layers;softmaxLayer('Name','Softmax')];
                layers = [layers;classificationLayer('Name','ClassOut')];
            end
        lgraph = layerGraph(layers);
        %% Add shortcut connections
        % Add shortcut connection around the convolutional units. Most shortcuts are 
        % identity connections.
        for i = 1:unitsPerStage-1
            lgraph = connectLayers(lgraph,['relu1' num2str(i)],['add1' num2str(i+1) '/in2']);
            lgraph = connectLayers(lgraph,['relu2' num2str(i)],['add2' num2str(i+1) '/in2']);
            lgraph = connectLayers(lgraph,['relu3' num2str(i)],['add3' num2str(i+1) '/in2']);
        end
        % Shortcut connection from input section to first stage. If unitType equals
        % "bottleneck", then the shortcut connection must upsample the channel
        % dimension from numFilters to numFilters*4.
        if this.unitType == "bottleneck"
            skip0 = [
                convolution2dLayer(1,numFilters*4,'Stride',1,'Name','skipConv0')
                batchNormalizationLayer('Name','skipBN0')];
            lgraph = addLayers(lgraph,skip0);
            lgraph = connectLayers(lgraph,'reluInp','skipConv0');
            lgraph = connectLayers(lgraph,'skipBN0','add11/in2');
        else
            lgraph = connectLayers(lgraph,'reluInp','add11/in2');
        end
        % Shortcut connection from stage one to stage two.
        if this.unitType == "bottleneck"
            numF =  numFilters*2*4;
        else
            numF =  numFilters*2;
        end
        skip1 = [convolution2dLayer(1,numF,'Stride',2,'Name','skipConv1')
            batchNormalizationLayer('Name','skipBN1')];
        lgraph = addLayers(lgraph,skip1);
        lgraph = connectLayers(lgraph,['relu1' num2str(unitsPerStage)],'skipConv1');
        lgraph = connectLayers(lgraph,'skipBN1','add21/in2');
        % Shortcut connection from stage two to stage three.
        if this.unitType == "bottleneck"
            numF =  numFilters*4*4;
        else
            numF =  numFilters*4;
        end
        skip2 = [convolution2dLayer(1,numF,'Stride',2,'Name','skipConv2')
            batchNormalizationLayer('Name','skipBN2')];
        lgraph = addLayers(lgraph,skip2);
        lgraph = connectLayers(lgraph,['relu2' num2str(unitsPerStage)],'skipConv2');
        lgraph = connectLayers(lgraph,'skipBN2','add31/in2');
        return
        end
        %% 

        function layers = standardConvolutionalUnit(~,filterSize,numF,stride,tag)
        layers = [
            convolution2dLayer(filterSize,numF,'Padding','same','Stride',stride,'Name',[tag,'conv1'])
            batchNormalizationLayer('Name',[tag,'BN1'])
            reluLayer('Name',[tag,'relu1'])
            convolution2dLayer(filterSize,numF,'Padding','same','Name',[tag,'conv2'])
            batchNormalizationLayer('Name',[tag,'BN2'])];
        end
        %% 

        function layers = bottleneckConvolutionalUnit(~,filterSize,numF,stride,tag)
        layers = [
            convolution2dLayer(1,numF,'Padding','same','Name',[tag,'conv1'])
            batchNormalizationLayer('Name',[tag,'BN1'])
            reluLayer('Name',[tag,'relu1'])

            convolution2dLayer(filterSize,numF,'Padding','same','Stride',stride,'Name',[tag,'conv2'])
            batchNormalizationLayer('Name',[tag,'BN2'])
            reluLayer('Name',[tag,'relu2'])

            convolution2dLayer(1,4*numF,'Padding','same','Name',[tag,'conv3'])
            batchNormalizationLayer('Name',[tag,'BN3'])];
        end

           
    end
end


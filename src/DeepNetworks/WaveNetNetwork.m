classdef WaveNetNetwork < DNNSuperClass
    
    properties
        pool_size_1 = 4;
        pool_size_2 = 8;

        n_filters=8;
        kernel_size=2;
        firstFilter=16;
        inputWidth;
    end
    
    methods
        function this = WaveNetNetwork(depth)
            this.depth  = depth;
            this.options = trainingOptions('adam','maxepoch',1);
        end
        
        function lgraph = buildGraph(this)
            this.inputWidth = this.inputDimention(2);
            this.lgraph = this.waveNetFunc();
            lgraph = this.lgraph;
        end
        
        function lgraph = waveNetFunc(this)
             if this.filter2D
                pool_size_1 = [ this.pool_size_1,  this.pool_size_1];
                pool_size_2 = [ this.pool_size_2,  this.pool_size_2];
            else
                pool_size_1 = [1,  this.pool_size_1];
                pool_size_2 = [1, this.pool_size_2];
            end    
            input_shape = [this.inputDimention 1];
            layers = [imageInputLayer(input_shape,'Name','input','Normalization',"zscore","NormalizationDimension","auto")
                      convolution2dLayer([1,this.firstFilter],this.n_filters,'Padding','same','Stride',1,'Name','conv1');
                      batchNormalizationLayer('Name','BN1');
                      reluLayer('Name','relu1')];
            for i = 1:this.depth
                if i~= 1
                     connections{i,1} = layers(end).Name;
                else
                     connections{i,1} = layers(4).Name;
                end

                layers = [layers;this.convolutionalUnit(this.n_filters,this.kernel_size^(i-1),['U',int2str(i)],this.kernel_size)];
                connections{i,2} = [layers(end).Name,'/in2'];
                skips{i} = layers(end-1).Name;

            end
            layers = [layers(1:end-1);
            additionLayer(i,'Name','addSkips')
            reluLayer('Name','reluAdder')


            convolution2dLayer(pool_size_1,this.n_filters,'Padding','same','Name',['final','conv1']);
            batchNormalizationLayer('Name',['final','BN1']);
            reluLayer('Name',['final','relu1']);
            averagePooling2dLayer(pool_size_1,'Stride',pool_size_1,'Name','finalPool1')
            convolution2dLayer(pool_size_2,this.n_filters,'Padding','same','Name',['final','conv2']);
            batchNormalizationLayer('Name',['final','BN2']);
            reluLayer('Name',['final','relu2']);
            convolution2dLayer(pool_size_2,this.n_filters,'Padding','same','Name',['final','conv3']);
            batchNormalizationLayer('Name',['final','BN3']);
            reluLayer('Name',['final','relu3']);
            averagePooling2dLayer(pool_size_2,'Stride',pool_size_2,'Name','finalPool2');

            convolution2dLayer([1,1],this.outputDimention,'Padding','same','Name',['final','conv4']);

            globalAveragePooling2dLayer('Name','globalPool')];

            if this.classification == true
                layers = [layers; softmaxLayer('Name','softmax')
                classificationLayer('Name','classOutput')];
            else
                layers = [layers;regressionLayer('Name','regOutput')];
            end
            lgraph = layerGraph(layers);
            for i = 1:length(connections)-1
                lgraph = connectLayers(lgraph,connections{i,1},connections{i,2});
            end

            for i = 1:length(skips)-1
                lgraph = connectLayers(lgraph,skips{i},['addSkips/in',int2str(i+1)]);
            end
        end


        function layers = convolutionalUnit(this,numF,dilation,tag,filter)
             if this.filter2D
                dilation = [dilation, dilation];
                filter = [filter, filter];
            else
                dilation = [1, dilation];
                filter = [1, filter];
            end    
            layers = [
                convolution2dLayer(filter,numF,'Padding','same','Stride',1,'DilationFactor',dilation,'Name',[tag,'conv1'])
                batchNormalizationLayer('Name',[tag,'BN1'])
                reluLayer('Name',[tag,'relu1'])
                convolution2dLayer([1,1],numF,'Padding','same','Name',[tag,'conv2'])
                batchNormalizationLayer('Name',[tag,'BN2'])
                additionLayer(2,'Name',[tag,'add'])
                ];
        end
    end
end


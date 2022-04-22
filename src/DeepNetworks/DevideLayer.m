classdef DevideLayer < nnet.layer.Layer
    properties
        sensorLens
    end
    
    methods
        function layer = DevideLayer(numOutputs,name,sensorLens) 
            layer.Name = name;
            layer.NumOutputs = numOutputs;
            layer.sensorLens =sensorLens;
        end
        
        function varargout  = predict(layer, X)
            start = 1;
            last = 0;
            for i = 1:layer.NumOutputs
                last = last+(layer.sensorLens(i));
                varargout {i} = X(:,start:last,:,:);
                start = last+1;
            end
        end
    end
end
classdef (Abstract) SupervisedUncertaintyTrainable < handle 
    
    methods
        this = train(this, data, target, U);
    end
end


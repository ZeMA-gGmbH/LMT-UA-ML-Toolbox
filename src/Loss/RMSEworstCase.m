classdef RMSEworstCase < handle
    %CLASSIFICATIONERROR Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods (Static)
        function l = loss(pred, target, unc)
            %root of mean squared error for worst case
            try
                upper_dist = abs(pred + unc - target);
                lower_dist = abs(pred - unc - target);
                tmp = max(upper_dist,lower_dist);
                l = sqrt(mean((tmp).^2));
            catch ME
                disp(ME);
                disp(ME.stack)
                l = Inf;
            end
        end
    end
end


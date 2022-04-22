classdef MultiSensorSingleExtractor < Appliable & UnSupervisedTrainable
    %MULTISENSOREXTRACTOR Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        extractionAlgs = {@BFCExtractor};
        extractors = {};
        extArgs = {};
        %Percent of feat out
        percentOut = true;
        numF = {{100}};
    end
    
    methods
        function this = MultiSensorSingleExtractor(extractionAlgs, extArgs,percentOut,numF)
            if nargin > 0
                if exist('extractionAlgs', 'var') && ~isempty(extractionAlgs)
                    this.extractionAlgs = extractionAlgs;
                end
                if exist('extArgs', 'var') && ~isempty(extArgs)
                    this.extArgs = extArgs;
                end
                if exist('percentOut', 'var') && ~isempty(percentOut)
                    this.percentOut = percentOut;
                end
                if exist('numF', 'var') && ~isempty(numF)
                    this.numF = numF;
                end
            end
        end
        
        function infoCell = info(this)
            infoCell = [];
            for lv = 1:size(this.extractors,1)
                for i = 1:size(this.extractors,2)
                    temp = this.extractors{lv,i}.info();
                    if this.percentOut == true
                            if size(this.numF,1) == 1 && size(this.numF,2) == 1
                                temp = temp(1:ceil(size(temp,1)*this.numF{1,1}{1}/100),:);
                            elseif size(this.numF,1) == 1 && size(this.numF,2) >= 1 
                                temp = temp(1:ceil(size(temp,1)*this.numF{1,i}{1}/100),:);
                            else
                                temp = temp(1:ceil(size(temp,1)*this.numF{lv,i}{1}/100),:);
                            end
                        else
                            if size(this.numF,1) == 1 && size(this.numF,2) == 1
                                temp = temp(this.numF{1,1}{1},:);
                            elseif size(this.numF,1) == 1 && size(this.numF,2) >= 1 
                                temp = temp(this.numF{1,i}{1},:);
                            else
                                temp = temp(this.numF{lv,i}{1},:);
                            end
                        end
                    temp1 = cell(size(temp,1),1);
                    for j = 1:size(temp,1)
                        temp1{j} = i;
                    end
                    while size(infoCell,2)>size([temp temp1],2)
                        temp1 = [temp1 temp1];
                    end
                    infoCell = [infoCell; temp temp1];
                end
            end
        end
        
        function train(this, data)
            %data is cell array of matrices
            %ToDo: check all matrices for the same number of rows
            
            %also allow single sensor extraction
            if ~iscell(data)
                data = {data};
            end
            
            this.extractors = cell(size(this.extractionAlgs,2),1);
            for j = 1:size(this.extractionAlgs,2)
                arguments = this.extArgs{j};
                this.extractors{j} = feval(this.extractionAlgs{j}, arguments{:});
                if isa(this.extractors{j}, 'UnSupervisedTrainable')
                    this.extractors{j}.train(data{j});
                end
            end
        end
        
        function feat = apply(this,data)
            %also allow single sensor extraction
            if ~iscell(data)
                data = {data};
            end
            featOut = [];
            for j = 1:size(this.extractionAlgs,2)
                feat = cell(size(data));

                temp = this.extractors{j}.apply(data{j});
                if this.percentOut == true
                    if size(this.numF,1) == 1 && size(this.numF,2) == 1
                        feat{j} = temp(:,1:ceil(size(temp,2)*this.numF{1}{1}/100));
                    elseif size(this.numF,1) == 1 && size(this.numF,2) >= 1 
                        feat{j} = temp(:,1:ceil(size(temp,2)*this.numF{1}{1}/100));
                    else
                        feat{j} = temp(:,1:ceil(size(temp,2)*this.numF{j}{1}/100));
                    end
                else
                    if size(this.numF,1) == 1 && size(this.numF,2) == 1
                        feat{j} = temp(:,this.numF{1,1}{1});
                    elseif size(this.numF,1) == 1 && size(this.numF,2) >= 1 
                        feat{j} = temp(:,this.numF{1}{1});
                    else
                        feat{j} = temp(:,this.numF{j}{1});
                    end
                end
                
                feat = horzcat(feat{:});
                featOut = [featOut feat];
            end
            feat = featOut;
        end
    end
end
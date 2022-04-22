classdef SimpleTrainingStackUncertainty < SupervisedUncertaintyTrainable & Appliable & Uncertainty
    
    properties
        stack = {};
        obj = {};
        args = [];
    end
    
    methods
        function this = SimpleTrainingStackUncertainty(s, args)
            if nargin > 1
                if exist('s', 'var') && ~isempty(s)
                    this.stack = s;
                end
                if exist('args', 'var') && ~isempty(args)
                    this.args = args;
                end
            end
        end
        
        function train(this, data, target, U)
            this.obj = cell(size(this.stack));
            for i = 1:length(this.stack)
                if ~isempty(this.args)
                    arguments = this.args{i};
                    this.obj{i} = feval(this.stack{i}, arguments{:});
                else
                    this.obj{i} = feval(this.stack{i});
                end
            end
            
            if nargin < 3
                %ToDo: confirm that all obj are unsupervised trainable or
                %do not need training. Otherwise throw error
            end
            
            for i = 1:length(this.stack)
                if isa(this.obj{i}, 'SupervisedTrainable')
                    this.obj{i}.train(data, target);
                elseif isa(this.obj{i}, 'UnSupervisedTrainable')
                    this.obj{i}.train(data);
                elseif isa(this.obj{i}, 'SupervisedUncertaintyTrainable')
                    this.obj{i}.train(data, target, U);
                elseif ~isa(this.obj{i}, 'Appliable')
                    throw('invalid Object in SimpleTrainingStack')
                end
                
                data_tmp = data;
                data = this.obj{i}.apply(data);
                
                if isa(this.obj{i}, 'Uncertainty')
                    U = this.obj{i}.uncertainty(U,data_tmp); 
                else
                    U = U;
                end
            end
        end
        
        function results = apply(this, data)
            %ToDo: Confirm this is trained
            for i = 1:length(this.stack)
                data = this.obj{i}.apply(data);
            end
            results = data;
        end
        
        function infoCell = info(this)
            infoCell = [];
            firstFlag = false;
            for i = 1:size(this.obj,2)
                if(ismethod(this.obj{1,size(this.obj,2)+1-i},"info"))
                    if(firstFlag == true)
                        infoCellTemp1 = this.obj{size(this.obj,2)+1-i}.info();
                        if(size(infoCellTemp1,2)>2)
                            infoCellTemp2 = [infoCell cell(size(infoCell,1),size(infoCellTemp1,2)-2)];
                        else
                            infoCellTemp2 = infoCell;
                        end
                        %k = Feature
                        for k = 1:size(infoCellTemp2,1)
                            infoCellTemp2{k,1} = [];
                            infoCellTemp2{k,2} = [];
                            for j = 1:size(infoCell(k,2),1)
                                infoCellTemp2{k,2} = [infoCellTemp2{k,2} infoCellTemp1{infoCell{k,2}(j),2}];
                                infoCellTemp2{k,1} = [infoCellTemp2{k,1} infoCell{k,1}(j)*infoCellTemp1{infoCell{k,2}(j),1}];
                                if(size(infoCellTemp1,2)>2)
                                    for lv=1:size(infoCellTemp1,2)-2
                                        infoCellTemp2{k,lv+2} = [infoCellTemp2{k,lv+2} infoCellTemp1{infoCell{k,2}(j),lv+2}];
                                    end
                                    
                                end
                            end
                        end
                        infoCell = infoCellTemp2;
                        if(size(infoCellTemp1,2)>2)
                            break;
                        end

                    else
                        infoCell = this.obj{size(this.obj,2)+1-i}.info();    
                    end
                    firstFlag = true;
                else
                    infoCell = [];
                    firstFlag = false;
                end
            end
        end
        
        function show(this)
            for i = 1:size(this.obj,2)
                if ismethod(this.obj{1,i},"show")
                    this.obj{1,i}.show();
                end
            end
        end
        
        function showAxisData(this)
            for i = 1:size(this.obj,2)
                if ismethod(this.obj{1,i},"showAxisData")
                    this.obj{1,i}.showAxisData();
                end
            end
        end
        
        function showUncertainty(this,data,U)
            for i = 1:size(this.obj,2)
                U = this.obj{1,i}.uncertainty(U,data);
                data = this.obj{1,i}.apply(data);
                if ismethod(this.obj{1,i},"showUncertainty")
                    this.obj{1,i}.showUncertainty(U,data);
                end
            end
        end
        
        function showUncertaintyAxisData(this,data,U)
            for i = 1:size(this.obj,2)
                U = this.obj{1,i}.uncertainty(U,data);
                data = this.obj{1,i}.apply(data);
                if ismethod(this.obj{1,i},"showUncertaintyAxisData")
                    this.obj{1,i}.showUncertaintyAxisData(U,data);
                end
            end
        end
        
        function showUncWithResults(this,data,U)
            if ismethod(this.obj{1,4},"showUncertaintyAxisData")
            	this.obj{1,4}.showUncertaintyAxisData(U,data);
            end
        end

        function U = uncertainty(this, U_x, data)
            if iscell(data)
                U_tmp = cell(1,size(data,2));
            else
                U_tmp = cell(1,1);
            end
            
            for i = 1 : size(U_tmp,2)
                if iscell(data)
                    U_tmp{1,i} = U_x{1,i};
                else
                    U_tmp{1,i} = U_x;
                end
            end
            
            for i = 1:size(this.obj,2)
                if ismethod(this.obj{1,i},"uncertainty")
                    if iscell(data)
                        U_tmp{1,1} = this.obj{1,i}.uncertainty(U_tmp, data);
                        data = this.obj{1,i}.apply(data);
                    else
                        U_tmp{1,1} = this.obj{1,i}.uncertainty(U_tmp{1,1}, data);
                        data = this.obj{1,i}.apply(data);
                    end
               end
            end
            U = U_tmp{1,1};
        end
    end
end


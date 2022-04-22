classdef constantSincLayer < nnet.layer.Layer
%   This layer is shipped as a part of the Speaker Identification using
%   Custom SincNet Layer and Deep Learning example and is not a toolbox
%   layer. This layer might be changed in a future release

%   Copyright 2020-2021 The MathWorks, Inc.

    properties (Access = private)
        % Number of sincNet filters in the sincNet layer
        NumFilters;
        % Sampling frequency 
        SampleRate;
        % Length of the sincNet filter
        FilterLength;
        % Number of channels in the speech signal
        NumChannels;
        % Time domain filters
        Filters;
        % Minumum starting frequency for the bandpass filters
        MinimumFrequency;
        % Minimum bandwidth for the bandpass filters
        MinimumBandwidth;
        % Starting frequencies of the bandpass filter
        StartFrequencies;
        % Bandwidth of the band pass filter
        Bandwidths;
    end
    
    methods
        function layer = constantSincLayer(NumFilters, FilterLength, SampleRate, NumChannels, Name)

            layer.NumFilters = NumFilters;
            layer.FilterLength = FilterLength;
            layer.SampleRate = SampleRate;
            layer.MinimumFrequency = 50.0;
            layer.MinimumBandwidth = 50.0;
            
            % Mel Initialization of the filterbanks
            low_freq_mel = 80;
            high_freq_mel = hz2mel(SampleRate/2 - layer.MinimumFrequency - layer.MinimumBandwidth);  
            mel_points = linspace(low_freq_mel,high_freq_mel,NumFilters); 
            f_cos = mel2hz(mel_points);
            b1 = circshift(f_cos,1);
            b2 = circshift(f_cos,-1);
            
            b1(1) = 30; % Min b1 of filter = 30 Hz
            b2(end) = (SampleRate/2) - 100;
            
            N = layer.FilterLength;
            time_stamps = linspace(-(N-1)/2,(N-1)/2,N)/layer.SampleRate;
            
            if mod(N,2) == 1
                time_stamps(round(N/2)) = eps;
            end
            
            % Hamming Window
            n = linspace(0,N,N);
            window = 0.54 - 0.46*cos(2*pi*n/N);
            
            % sincNet layer learnable parameters
            layer.StartFrequencies = b1/SampleRate;
            layer.Bandwidths  = (b2 - b1)/SampleRate;
            
            % Set layer name.
            layer.Name = Name;
            
            % Set layer description.
            layer.Description = "ConstantSincLayer with " + NumChannels + " channels";
            
            % Computing the Constant Sinc Filter Bank
            filt_beg_freq = abs(layer.StartFrequencies) + layer.MinimumFrequency/layer.SampleRate;
            filt_end_freq = filt_beg_freq+(abs(layer.Bandwidths)+layer.MinimumBandwidth/layer.SampleRate);
            
            % Define Filter values
            low_pass1 = 2*filt_beg_freq.*sin(2*pi*layer.SampleRate*time_stamps'*filt_beg_freq)...
                                    ./(2*pi*layer.SampleRate*time_stamps'*filt_beg_freq);
            low_pass2 = 2*filt_end_freq.*sin(2*pi*layer.SampleRate*time_stamps'*filt_end_freq)...
                                    ./(2*pi*layer.SampleRate*time_stamps'*filt_end_freq);                
            band_pass = low_pass2 - low_pass1;
            band_pass = band_pass/max(band_pass(:));
            layer.Filters = single(reshape(window'.*band_pass,...
                    1,layer.FilterLength,1,layer.NumFilters,1));
                
        end
        
        function Z = predict(layer, X)
            % Convolve input with Sinc parametrized bandpass filters
            Z = dlconv(X,layer.Filters,0,'DataFormat','SSCB');
        end
        
        function plotNFilters(layer,n)
            % This layer plots n filters with equally spaced filter indices
            [H,Freq] = freqDomainFilters(layer);
            idx = linspace(1,layer.NumFilters,n);
            idx = round(idx);
            for jj = 1:n
                subplot(ceil(sqrt(n)),ceil(n/ceil(sqrt(n))),jj);
                plot(Freq(:,idx(jj)),H(:,idx(jj)));
                sgtitle("Filter Frequency Response")
                xlabel("Frequency (Hz)")
            end
        end
    end
    
    methods (Access = private)
        function [H,Freq] = freqDomainFilters(layer)
            % This method returns the frequency domain representation of
            % the Sinc parametrized bandpass filters
            F = squeeze(layer.Filters);
            H = zeros(size(F));
            Freq = zeros(size(F));
            for ii = 1:size(F,2)
                [h,f] = freqz(F(:,ii),1,layer.FilterLength,layer.SampleRate);
                H(:,ii) = abs(h);
                Freq(:,ii) = f;
            end
        end
    end
end
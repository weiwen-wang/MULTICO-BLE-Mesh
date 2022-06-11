import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct
import pandas as pd
from sklearn.preprocessing import normalize
import pyaudio
import wave
import time
import os

ll = np.arange(1,100001,1)   #used in the file name
ll = ','.join(str(i) for i in ll)
comptime = [] #record the computation time for each feature vector
fisize = []  # record the file memory footprint of each feature vector
nowt = []  # timestamp of each vector 


for ir in range(100000):
    now = time.time()
    nowt.append(now)
    print(now)
    form_1 = pyaudio.paInt16 # 16-bit resolution
    chans = 1 # 1 channel
    samp_rate = 44100 # 44.1kHz sampling rate, can change depanding on the microphone
    chunk = 4096 # 2^12 samples for buffer
    record_secs = 7 # seconds to record
    dev_index = 2 # device index found by p.get_device_info_by_index(ii)
    wav_output_filename = 'test.wav' # name of .wav file
    #print(wav_output_filename)
    audio = pyaudio.PyAudio()

    stream = audio.open(format = form_1,rate = samp_rate,channels = chans, \
                            input_device_index = dev_index,input = True, \
                            frames_per_buffer=chunk)
    print("recording")
    frames = []
    for ii in range(0,int((samp_rate/chunk)*record_secs)):
        data = stream.read(chunk)
        frames.append(data)
    print("finished recording")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    wavefile = wave.open(wav_output_filename,'wb')
    wavefile.setnchannels(chans)
    wavefile.setsampwidth(audio.get_sample_size(form_1))
    wavefile.setframerate(samp_rate)
    wavefile.writeframes(b''.join(frames))
    wavefile.close()

    
    start = time.process_time()
    audio_path = '/home/pi/audioiden/test.wav'  # voice sample, will overlap every time
    sample_rate,original_signal= scipy.io.wavfile.read(audio_path)
    pre_emphasis = 0.97
    emphasized_signal = np.append(original_signal[0], original_signal[1:] - pre_emphasis * original_signal[:-1])
    emphasized_signal_num = np.arange(len(emphasized_signal))

    # framing 
    frame_size = 0.025
    frame_stride = 0.01
    frame_length = int(round(frame_size*sample_rate))
    frame_step = int(round(frame_stride*sample_rate)) 
    signal_length = len(emphasized_signal)
    num_frames = int(np.ceil(float(np.abs(signal_length-frame_length))/frame_step))

    pad_signal_length = num_frames * frame_step + frame_length
    pad_signal = np.append(emphasized_signal, np.zeros((pad_signal_length - signal_length)))

    indices = np.tile(np.arange(0,frame_length),(num_frames,1))+np.tile(np.arange(0,num_frames*frame_step,frame_step), (frame_length, 1)).T
    frames = pad_signal[np.mat(indices).astype(np.int32, copy=False)]

    # haming window
    # W(m,a)=(1-a)- a*cos(2*pi*n)/(N-1), a = 0.46
    #N_w = 200
    #x = np.arange(N_w)
    #y = 0.54 * np.ones(N_w) - 0.46 * np.cos(2*np.pi*x/(N_w-1))

    frames *= np.hamming(frame_length)
    #print("(Nw window number, length of signal window)",frames.shape)

    # FFT
    NFFT = 512  #Nbins = 512 
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = (1.0 / NFFT) * (mag_frames ** 2)
    #print("Pw shape(Nw,bins/2)", pow_frames.shape)

    #Mel filter
    low_freq_mel = 0
        
    nfilt = 100
    #mel0scale frequency
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz

    bin = np.floor((NFFT + 1) * hz_points / sample_rate)
    
    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))

    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    #print("Ps shape(Nw,Ncep)",filter_banks.shape)
    filter_banks = 20 * np.log10(filter_banks)  # dB


    # Discrete Cosine Transform
    ## number of cepstral
    num_ceps = 100
    # remove the first column
    Cp = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)]
    (nframes, ncoeff) = Cp.shape
    # mean of each row 
    ave = np.average(Cp,axis = 1)
    Cave = np.repeat(ave,ncoeff, axis =0).reshape(nframes,ncoeff)
    C = Cp -Cave
    #print("Cshape(Nw,Ncep-1)",C.shape)


    #de-emphized by smoothening vector M, M=lift  ->Cs
    n = np.arange(ncoeff)
    cep_lifter =num_ceps -1
    M = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)    
    Cs = C * M
    #print("Cs shape",Cs.shape)

    #seperate by comparing normalized average -> Csp
    CN = np.average(Cs,axis = 1)
    CN_normed = (CN) / (CN).max(axis=0)
    muCN = np.average(CN_normed)


    # Csp separated from noise, if CN>muCN
    ver = []
    for t in range(len(CN_normed)):
        if CN_normed[t] < muCN:
            ver.append(t)

    #Csp contains only the selected column
    Csp = np.delete(Cs,ver,axis=0)
    Ccep = np.average(Csp,axis=0)


    # calculate first and second derivative
    C_delta = librosa.feature.delta(Cs,order = 1)
    C_delta2 =librosa.feature.delta(Cs,order = 2)

    # same procedure as Cep
    CN_d1 = np.average(C_delta,axis = 1)
    CN_normed_d1 = (CN_d1) / (CN_d1).max(axis=0)
    muCN_d1 = np.average(CN_normed_d1)
    ver_d1 = []
    for p in range(len(CN_normed_d1)):
        if CN_normed_d1[p] < muCN_d1:
            ver_d1.append(p)
    Csp_d1 = np.delete(C_delta,ver_d1,axis=0)
    Ccep_d1 = np.average(Csp_d1,axis=0)

    CN_d2 = np.average(C_delta2,axis = 1)
    CN_normed_d2 = (CN_d2) / (CN_d2).max(axis=0)
    muCN_d2 = np.average(CN_normed_d2)
    ver_d2 = []
    for q in range(len(CN_normed_d2)):
        if CN_normed_d2[q] < muCN_d2:
            ver_d2.append(q)
    Csp_d2 = np.delete(C_delta2,ver_d2,axis=0)
    Ccep_d2 = np.average(Csp_d2,axis=0)

    Fs=np.hstack((Ccep,Ccep_d1))
    Fs=np.hstack((Fs,Ccep_d2))
    #print("Fs elements number = 3(Ncep-1)=",len(Fs))

    dataframe = pd.DataFrame(Fs)
    dataframe.to_csv('/home/pi/audioiden/mfcc/'+ll[ir]+'.csv',index=False,sep=',')
                
    end = time.process_time()
    protime = start - end
    comptime.append(protime)
    start = 0
    end = 0
    #print("Comtupation time:%s Seconds" %(end-start))
    fsize = os.path.getsize('/home/pi/audioiden/mfcc/'+ll[ir]+'.csv')
    fisize.append(fsize)
    #print("Fs matrix file memory:%s Byte" %fsize)
    
    para = np.vstack((nowt,comptime))
    para = np.vstack((para,fisize))
    dataframe = pd.DataFrame(para)
    dataframe.to_csv('/home/pi/audioiden/testpara.csv',index=False,sep=',')
    print('success!')
    
    
    
    

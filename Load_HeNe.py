#%%
import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use ('Qt5Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
import matplotlib.collections as mcol
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from matplotlib.markers import TICKLEFT, TICKRIGHT, TICKUP, TICKDOWN, CARETLEFT, CARETRIGHT, CARETUP, CARETDOWN
from scipy import signal, interpolate, optimize
from collections import OrderedDict

#-------def---******************************************************************
def readcsv(datafilename):
    #csvの読み込みを行う関数
    df=pd.read_csv(datafilename,header=None) #datafilenameは〜.csv
    dft = np.array(df).T #読み込みしたcsvを横方向に転置
    dftr = dft[:,1:] #最初の文字列の行は取っ払う.文字列行ない場合はこの行をコメントアウト
    dftru = np.array(dftr,dtype=np.float64) #読み込んだ時点でstrだから，floatに直す!!これはめっちゃ重要!!
    return dftru #配列で返してるよ

def loadtxt(datafilename,skip_rows,delim):
    #csvの読み込みを行う関数，ロードしたいdatafilename(拡張子前)とskipしたい行数を右側に入れてね
    a = "./" + datafilename + ".txt" 
    df = np.loadtxt(a,skiprows = skip_rows, delimiter = delim )
    dftr = df.T
    return dftr

def Load_HeNe(datafilename):
    df = loadtxt(datafilename,5,",")
    fringe=df[1]
    #smoothing----------------------------------------------#
    x_org = np.linspace(0,(len(fringe)-1),len(fringe))
    x_aft = np.linspace(0,(len(fringe)-1),100000)
    fringe_interp = interpolate_akm(fringe,x_org,x_aft)
    fringe_ret = interpolate_akm(fringe_interp,x_aft,x_org)
    #-------------------------------------------------------#
    #physical constants-------------------------------------#
    HeNe_hacho = 6328e-10
    hikari = 2.99792458e8
    #-------------------------------------------------------#
    #fringe analysis----------------------------------------#
    fringe = fringe_ret
    wn = np.zeros(50000, dtype = float) 
    k = 0
    for i in range(0,(len(fringe)-1)):
        if (fringe[i]>=fringe[i+1] and fringe[i]*fringe[i+1]<=0.0):
            wn[k] = i
            k = k + 1
    wnc = np.trim_zeros(wn)
    #-------------------------------------------------------#
    #fringe interpolate-------------------------------------#
    x = np.linspace(0,(len(wnc)-1),len(wnc))
    x_interp = np.linspace(0,len(fringe),10000)
    wncx = HeNe_hacho/hikari * x * 1e12
    fringe_pos = interpolate_akm(wncx,wnc,x_interp)
    #-------------------------------------------------------#
    return fringe_pos

def interpolate_akm(graph,point_before,point_after):
    #グラフの点数を補間する．この関数では前データ点数情報(ex.delay(10000))からほしい後データ点数(ex.delayFT(8192))情報を入力すれば良い
    f = interpolate.Akima1DInterpolator(point_before, graph)
    y = f(point_after)
    return y

def delay_recalibration(delay,points):
    #fringeから得られたdelayは点数間隔が一定でないから，このままでは信号をフーリエ変換ができない．
    #したがって，点数間隔が一定なdelayを作って信号を更正する必要がある．．
    #(例えば)10000の間隔が一定でないデータを線形な8192のdelayに更正する関数である．
    points = int(points) #pointsはほしいデータ点数これをintにしとく
    x = np.linspace(0,(points - 1),points) #ほしいデータ点数個の整数をずらっと並べる
    length = len(delay) #元delayの点数
    length = int(length) #intにしとく
    delay_r = x * (delay[(length - 1)] - delay[0])/points + delay[0] #一次関数にしちゃうのだぜ
    return delay_r

def frqaxis_maker(delay,points):
    #周波数軸メーカー．元delayを入れて，補間したsignalの点数の半分の値をpointsにすれば良い．
    #これも点数間隔を一定にする家庭も含んでいる．
    #なぜなら点数間隔を一定にした信号をフーリエ変換したら，点数間隔が一定なFTスペクトルができるはずだから横軸も点数間隔が一緒でなくてはならない．
    points = int(points)
    x = np.linspace(0,(points - 1),points)
    length = len(delay)
    length = int(length)
    frq = x / (delay[(length - 1)] - delay[0]) * (2*points - 1)/(2*points)
    return frq

def FFT_abs_amp(Interpsig):
    #FFTの各要素複素数の大きさを取ったスペクトルを表したもの．
    #要素数は半分になるよ．
    #frqと要素数合う(はずだ)よ．
    #これは2乗が取れてるバージョン．igorは magnitude squared
    N = len(Interpsig)
    FFTsig = np.fft.fft(Interpsig)
    FFTsig_abs = np.abs(FFTsig)
    FFTsig_abs_amp = FFTsig_abs/N *2
    FFTsig_abs_amp[0] =  FFTsig_abs_amp[0]/2
    FFTsig_abs_amp_halfpoints = FFTsig_abs_amp[:int(N/2)]
    return FFTsig_abs_amp_halfpoints

def FFT_abs_display(FFTsignal,frqaxis):
    FFTsignal_len = len(FFTsignal)
    frqaxis_len = len(frqaxis)
    ymax = FFTsignal.max()
    if FFTsignal_len == frqaxis_len :
        plt.plot(frqaxis,FFTsignal)
        plt.xlim(0,8.5)
        plt.ylim(0,ymax*1.1)
        plt.legend()
        plt.show() 
    else:
        print('both of element numbers are not corresponding!!')

def FFT_abs_amp_series(row_sig,row_delay):
    if len(row_sig) == 10000:
        if len(row_delay) == 10000:
            delayFT = delay_recalibration(row_delay,8192)
            frq  = frqaxis_maker(row_delay,4096)
            Interpsig = interpolate_akm(row_sig,row_delay,delayFT) * 1e6 #1e6倍にしてるので，縦軸に(10^{-6})入れてね．
            Interpsig_FFT_abs_amp = FFT_abs_amp(Interpsig)  #1e6倍にしてるので，縦軸に(10^{-6})入れてね．
        else:
            print('Number of delay array is not 10000.')
    else:
        print('Number of signal array is not 10000.')
    
    return Interpsig, Interpsig_FFT_abs_amp, delayFT, frq

def index_seek(value,ary):
    ary_length = len(ary)
    s = ary[0]
    ind = 0
    for i in range(1,ary_length):
        if (np.abs(s - value)) >= (np.abs(ary[i] - value)):
            s = ary[i]
            ind = i
    return ind

def peak_seek(frq,sig,peak_frq):
    peakfrq_len = len(peak_frq)
    peak_inf = np.zeros((peakfrq_len,2))
    index_shift = index_seek(0.12,frq)
    for i in range(peakfrq_len):
        index_frq = index_seek(peak_frq[i],frq)
        start = np.abs(index_frq - index_shift)
        end = np.abs(index_frq + index_shift)
        peak_max = 0.0
        peak_pos = 0.0

        for j in range(start,end):
            if peak_max < sig[j] :
                peak_max = sig[j]
                peak_pos = frq[j]

        peak_inf[i][0] = peak_pos
        peak_inf[i][1] = peak_max
    
    return peak_inf

def peak_seek_simple(frq,sig,peak_frq):
    index_shift = index_seek(0.12,frq)
    index_frq = index_seek(peak_frq,frq)
    start = np.abs(index_frq - index_shift)       
    end = np.abs(index_frq + index_shift)
    peak_max = 0.0
    peak_pos = 0.0

    for j in range(start,end):
        if peak_max < sig[j] :
            peak_max = sig[j]
            peak_pos = frq[j]

    return peak_max, peak_pos
     
def window_function(position,width,delayFT):
    Anorm = 1.0/(np.sqrt(np.pi * 2.0 * width))
    y_win = Anorm * np.exp(-((delayFT - position)**2.0/(width)**(2.0)))
    return y_win

def STFT_origin(Interpsig,delayFT,window_width,start,end):
    start_index = index_seek(start,delayFT)
    end_index = index_seek(end,delayFT)
    total_index = (end_index - start_index) + 1
    data_len = int(len(delayFT)/2)
    y_fftabs = np.zeros((total_index,data_len),dtype = float)
    j = 0

    for i in range(start_index,end_index):
        y_win = window_function(delayFT[i],window_width,delayFT)
        y_conv = Interpsig * y_win
        y_fftabs[j] = FFT_abs_amp(y_conv)
        j += 1 

    return y_fftabs, start_index, end_index

def STFT_nrml(Interpsig,delayFT,window_width,start,end):
    start_index = index_seek(start,delayFT)
    end_index = index_seek(end,delayFT)
    total_index = (end_index - start_index) + 1
    data_len = int(len(delayFT)/2)
    y_fftabs = np.zeros((total_index,data_len),dtype = float)
    j = 0

    for i in range(start_index,end_index):
        y_win = window_function(delayFT[i],window_width,delayFT)
        y_conv = Interpsig * y_win
        y_fftabs[j] = FFT_abs_amp(y_conv)
        j += 1 

    y_fftabs_nrml = y_fftabs/y_fftabs.max()
    return y_fftabs_nrml, start_index, end_index

def erf_modoki(x,pos,width):
    return 1.0/(1.0 + np.exp(-(x-pos)/width)) 

def kink(x,pos,width,intensity):
    return 4.0 * intensity*erf_modoki(x,pos,width) * (- erf_modoki(x,pos,width) + 1)
#-------def---******************************************************************

#%%
delay = Load_HeNe("C3fringe00001")
plt.plot(delay)


#%%

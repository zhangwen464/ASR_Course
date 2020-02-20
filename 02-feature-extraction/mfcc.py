import librosa
import numpy as np
from scipy.fftpack import dct
import pylab as plt
import math
import matplotlib as mplt

# If you want to see the spectrogram picture
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def plot_spectrogram(spec, note,file_name):
#     """Draw the spectrogram picture
#         :param spec: a feature_dim by num_frames array(real)
#         :param note: title of the picture
#         :param file_name: name of the file
#     """ 
     fig = plt.figure(figsize=(20, 5))
     heatmap = plt.pcolor(spec)
     fig.colorbar(mappable=heatmap)
     plt.xlabel('Time(s)')
     plt.ylabel(note)
     plt.tight_layout()
     plt.savefig(file_name)


#preemphasis config 
alpha = 0.97

# Enframe config
frame_len = 400      # 25ms, fs=16kHz
frame_shift = 160    # 10ms, fs=15kHz
fft_len = 512

# Mel filter config
num_filter = 23
num_mfcc = 12

# Read wav file
wav, fs = librosa.load('./test.wav', sr=None)

# Enframe with Hamming window function
def preemphasis(signal, coeff=alpha):
    """perform preemphasis on the input signal.

        :param signal: The signal to filter.
        :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.97.
        :returns: the filtered signal.
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def enframe(signal, frame_len=frame_len, frame_shift=frame_shift, win=np.hamming(frame_len)):
    """Enframe with Hamming widow function.

        :param signal: The signal be enframed
        :param win: window function, default Hamming
        :returns: the enframed signal, num_frames by frame_len array
    """
    
    num_samples = signal.size
    num_frames = np.floor((num_samples - frame_len) / frame_shift)+1
    frames = np.zeros((int(num_frames),frame_len))
    for i in range(int(num_frames)):
        frames[i,:] = signal[i*frame_shift:i*frame_shift + frame_len] 
        frames[i,:] = frames[i,:] * win

    return frames

def get_spectrum(frames, fft_len=fft_len):
    """Get spectrum using fft
        :param frames: the enframed signal, num_frames by frame_len array
        :param fft_len: FFT length, default 512
        :returns: spectrum, a num_frames by fft_len/2+1 array (real)
    """
    cFFT = np.fft.fft(frames, n=fft_len)
    valid_len = int(fft_len / 2 ) + 1
    spectrum = np.abs(cFFT[:,0:valid_len])
    # num_frames * fft_len/2+1
    return spectrum

def fbank(spectrum, num_filter = num_filter):
    """Get mel filter bank feature from spectrum
        :param spectrum: a num_frames by fft_len/2+1 array(real)
        :param num_filter: mel filters number, default 23
        :returns: fbank feature, a num_frames by num_filter array 
        DON'T FORGET LOG OPRETION AFTER MEL FILTER!
    """

    feats=np.zeros((int(fft_len/2+1), num_filter)) # 257* 23 
    """
        FINISH by YOURSELF
    """
    #min_fs=300hz,max_fs=8Khz 2595
    min_mel_fs = 2595 * math.log10(1+300/700)
    max_mel_fs = 2595 * math.log10(1+8000/700)

    w2 = int(fft_len/2+1)
    df = fs/fft_len
    freq = []
    for i in range(0,w2):
        freqs = int(i * df)
        freq.append(freqs)

    # num_filter = 23
    #average the mel_fs
    mel_fs_array = np.linspace(min_mel_fs, max_mel_fs, num_filter + 2)
    # mel to hz
    Hz_fs_array = 700.0 * ( 10**(mel_fs_array / 2595) - 1.0)
    print (Hz_fs_array)
    bin_array = np.floor(Hz_fs_array / df)
    print (bin_array)

    bank = np.zeros((num_filter, w2))
    for k in range(1,num_filter+1):
        f1 = bin_array[k-1]
        f2 = bin_array[k+1]
        f0 = bin_array[k]
        for i in range (1, w2):
            if i > f1 and i <= f0:
                bank[k-1,i] = (i-f1)/(f0-f1)
            elif i >f0 and i<=f2:
                bank[k-1,i] = (f2-i)/(f2-f0)
        #print (k)
        #print(bank[k-1,:])
        #plt.plot(freq,bank[k-1,:],'r')
    #plt.show()

    #feats(356,23)=mul((356,257)(257,23))
    spectrum_power = np.power(spectrum,2) / fft_len
    #feats = np.matmul(spectrum , np.transpose(bank))
    feats_power = np.matmul(spectrum_power , np.transpose(bank))
    feats = np.log(feats_power)
    #feats = np.transpose(feats)
    mplt.image.imsave('./out.png', feats)
    return feats

def mfcc(fbank, num_mfcc = num_mfcc):
    """Get mfcc feature from fbank feature
        :param fbank: a num_frames by  num_filter array(real)
        :param num_mfcc: mfcc number, default 12
        :returns: mfcc feature, a num_frames by num_mfcc array 
    """
    feats = np.zeros((fbank.shape[0],num_mfcc))
    feats= dct(fbank, type=2, axis=1, norm='ortho')[:,:num_mfcc]
    print("feat:",feats.shape)
    #plt.plot(feats)
    #plt.show()
    return feats

def write_file(feats, file_name):
    """Write the feature to file
        :param feats: a num_frames by feature_dim array(real)
        :param file_name: name of the file
    """
    f=open(file_name,'w')
    (row,col) = feats.shape
    for i in range(row):
        f.write('[')
        for j in range(col):
            f.write(str(feats[i,j])+' ')
        f.write(']\n')
    f.close()


def main():
    wav, fs = librosa.load('./test.wav', sr=None)
    signal = preemphasis(wav)
    frames = enframe(signal)
    spectrum = get_spectrum(frames)
    fbank_feats = fbank(spectrum)
    mfcc_feats = mfcc(fbank_feats)
    plot_spectrogram(fbank_feats, 'Filter Bank','fbank.png')
    write_file(fbank_feats,'./test.fbank')
    plot_spectrogram(mfcc_feats.T, 'MFCC','mfcc.png')
    write_file(mfcc_feats,'./test.mfcc')

if __name__ == '__main__':
    main()

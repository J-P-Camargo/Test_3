# Célula 1: imports e utilitários
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, find_peaks
from scipy.optimize import curve_fit
import random

FS = 48000
BLOCK = 4096
HOP = BLOCK // 2
BAND_MIN = 500
BAND_MAX = 18000

def compute_rho_from_signal(x, fs=FS, N=BLOCK, hop=HOP, band=(BAND_MIN,BAND_MAX), peak_thresh=6.0):
    f, t_seg, Zxx = stft(x, fs=fs, window='hann', nperseg=N, noverlap=N-hop, boundary=None)
    mag = np.abs(Zxx)
    band_mask = (f >= band[0]) & (f <= band[1])
    rhos = []
    # simplified "crista" tracking: pick peak freq at each time column
    for col in range(mag.shape[1]):
        mag_col = mag[:,col]
        band_col = mag_col[band_mask]
        if band_col.size == 0: continue
        k = np.argmax(band_col)
        idx = np.where(band_mask)[0][0] + k
        # estimate local frequency series by following nearest peaks across cols (toy)
        # for simplicity compute local slope over a short window of 6 frames
    # A simple implementation: estimate instantaneous freq by center of mass in band per frame
    inst_freq = []
    for col in range(mag.shape[1]):
        mag_col = mag[:,col]
        band_col = mag_col[band_mask]
        ff = f[band_mask]
        denom = band_col.sum() + 1e-12
        cf = (ff * band_col).sum()/denom
        inst_freq.append(cf)
    inst_freq = np.array(inst_freq)
    # compute tilt rho in sliding windows over frames (window length M frames)
    M = 25
    for i in range(0, len(inst_freq)-M+1):
        seg = inst_freq[i:i+M]
        tau = np.arange(M)
        x_ = seg - seg.mean()
        t_ = tau - tau.mean()
        denom = np.sqrt((x_**2).sum() * (t_**2).sum()) + 1e-12
        r = (x_*t_).sum()/denom
        rhos.append(r)
    return np.array(rhos)

def generate_signal_synthetic(A, dur=5.0, fs=FS, n_freqs=50, fmin=1000, fmax=15000, chirp_base=0.0):
    """
    Model: sum of n_freqs with chirp rate proportional to A (toy model linking A->G1).
    chirp_base allows an offset
    """
    t = np.linspace(0, dur, int(dur*fs), endpoint=False)
    freqs = np.linspace(fmin, fmax, n_freqs)
    phases = np.random.uniform(0,2*np.pi,size=n_freqs)
    sig = np.zeros_like(t)
    for i,f0 in enumerate(freqs):
        k = chirp_base + A  # make chirp proportional to A
        phase = 2*np.pi*(f0*t + 0.5*k*t**2) + phases[i]
        sig += (1.0/n_freqs) * np.sin(phase)
    # normalize
    sig /= np.sqrt(np.mean(sig**2)) + 1e-12
    return sig

def fit_linear(x,y):
    def lin(x,a,b): return a*x + b
    popt, pcov = curve_fit(lin, x, y)
    return popt, np.sqrt(np.diag(pcov))
############################################################################################################################
# Célula 2: sweep em A e rodar simulações
As = np.linspace(0, 1000, 9)  # intensidade de G1 simulada (arbitrária)
means = []
stds = []
Ntrials = 20
for A in As:
    trial_vals = []
    for s in range(Ntrials):
        sig = generate_signal_synthetic(A, dur=5.0, n_freqs=50, chirp_base=0.0)
        rhos = compute_rho_from_signal(sig)
        if len(rhos)==0:
            trial_vals.append(0.0)
        else:
            trial_vals.append(np.mean(np.abs(rhos)))
    means.append(np.mean(trial_vals))
    stds.append(np.std(trial_vals))
means = np.array(means); stds = np.array(stds)
###############################################################################################################################
# Célula 3: ajuste linear e plot
popt, perr = fit_linear(As, means)
a,b = popt; da, db = perr
plt.errorbar(As, means, yerr=stds, fmt='o', label='simulated')
xx = np.linspace(As.min(), As.max(), 200)
plt.plot(xx, a*xx+b, '-', label=f'fit: a={a:.3e} ± {da:.1e}')
plt.xlabel('A (sim proxy for <G1>)')
plt.ylabel('mean |rho|')
plt.title('Simulação: resposta linear esperada')
plt.legend()
plt.show()

print("fit params:", popt, "errors:", perr)



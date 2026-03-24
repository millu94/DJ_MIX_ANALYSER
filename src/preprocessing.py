import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

class Preprocessing:
    def __init__(self,audio_dir,labels_csv):
        self.audio_dir = audio_dir
        self.labels_csv = labels_csv

    def load_audio(self,audio_dir):
        """
        Load the audio files from the audio directory
        """
        audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
        return audio_files
    

    def zero_crossing_rate(x):
        """
        Compute the zero-crossing rate for a 1D audio signal.
        
        Parameters
        ----------
        x : NumPy Array
            Audio signal. Shape (n_samples,).
            
        Returns
        -------
        float
            Zero-crossing rate (range [0, 1]).
        """
        x = np.asarray(x, dtype=float)

        x_sign = np.where(x>0, 1, 0)
        x_diff = np.diff(x_sign)
        return np.mean(np.abs(x_diff))
    
        """
        DIRECTLY TAKEN FROM AML NOTEBOOK
        """
    def spectral_centroid(x, sr):
        """
        Compute the spectral centroid of an audio signal, in Hz.
        The centroid is computed once for the entire signal.

        Formula:
            centroid = sum(f_k * |X_k|) / sum(|X_k|)
        where X_k is the FFT and f_k are bin center frequencies. |X| denotes the absolute of X

        Args:
            x : NumPy Array
                Audio signal. Shape (n_samples,).
            sr (int or float): Sampling rate in Hz.

        Returns:
            float: Spectral centroid in Hz.

        DIRECTLY TAKEN FROM AML NOTEBOOK
        """
        print("Running spectral_centroid")
        # Real FFT and frequency bins
        X = np.fft.rfft(x)
        f = np.fft.rfftfreq(x.shape[0], d=1.0 / sr)

        # Take absolute
        X_abs = np.abs(X)

        # Find denominator
        denom = X_abs.sum()

        if denom <= 1e-12: # If input is silent, return 0
            return 0.0 
        
        # Calculate spectral centroid
        centroid_hz = (f * X_abs).sum() / denom
        return centroid_hz
    def onset_strength_envelope(x, sr):
        """
        Compute the onset strength envelope of an audio signal.
        The onset strength envelope is a measure of the energy at each time step.
        """
        print("Running onset_strength_envelope")
        return librosa.onset.onset_strength(y=x, sr=sr)
    def tempogram(x, sr):
        """
        Compute the tempogram of an audio signal.
        The tempogram is a measure of the tempo at each time step.
        """
        print("Running tempogram")
        return librosa.feature.tempogram(y=x, sr=sr)
    def tempo(x, sr):
        """
        Compute the tempo of an audio signal.
        The tempo is a measure of the tempo of the audio signal.
        """
        print("Running tempo")
        # librosa returns an array; we reduce it to a scalar so any downstream
        # "std" feature would be redundant.
        return float(np.mean(librosa.beat.tempo(y=x, sr=sr)))
    def chroma_stft(x, sr):
        """
        Compute the chroma stft of an audio signal.
        The chroma stft is a measure of the chroma at each time step.
        """
        print("Running chroma_stft")
        return librosa.feature.chroma_stft(y=x, sr=sr)
    def chroma_cqt(x, sr):
        """
        Compute the chroma cqt of an audio signal.
        The chroma cqt is a measure of the chroma at each time step.
        """
        print("Running chroma_cqt")
        return librosa.feature.chroma_cqt(y=x, sr=sr)
    def chroma_cens(x, sr):
        """
        Compute the chroma cens of an audio signal.
        The chroma cens is a measure of the chroma at each time step.
        """
        print("Running chroma_cens")
        return librosa.feature.chroma_cens(y=x, sr=sr)
    def mel_frequency_cepstral_coefficients(x, sr):
        """
        Compute the mel frequency cepstral coefficients of an audio signal.
        The mel frequency cepstral coefficients are a measure of the mel frequency cepstral coefficients at each time step.
        """
        print("Running mel_frequency_cepstral_coefficients")
        return librosa.feature.mfcc(y=x, sr=sr)

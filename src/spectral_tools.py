"""
.. module:: spectral_tools
   :platform: Windows
   :synopsis: Spectral editing tools for Deharm application

.. moduleauthor:: Janne Valtanen (janne.valtanen@infowader.com)
"""

import tempfile
import subprocess
import os

import numpy as np
import scipy.io.wavfile as wavfile
from scipy.fftpack import rfft, irfft
from scipy.signal import resample, stft, istft
import lameenc
import librosa
from librosa.effects import hpss


def get_audio_data(filename):
    """Get audio data based on a filename

    Args:
        filename(str): Name of the file

    Returns:
        (tuple): tuple containing:
            audio_data(numpy.ndarray): audio data in a NumPy array
            sfreq(float): Sampling frequency in Hz
            nof_channels(int): Number of channels
            length_s(int): Audio length in samples
            length_t(float): Audio length in seconds

    Raises:
        Exception
    """
    # FIXME: Only MP3 and WAVs have been tested!
    if filename.lower().endswith(".mp3"):
        # Use FFMPEG directly to decode MP3.
        # Using librosa to decode MP3 without a console will crash.
        temp_audiofile = tempfile.NamedTemporaryFile(suffix=".wav",
                                                     delete=False)
        temp_audiofile.close()
        # FIXME: Sample rate is fixed to 44.1kHz
        try:
            # Shell needs to True for Windows for the standalone executable
            shell = (os.name == "nt")

            p = subprocess.Popen(['ffmpeg',
                                  '-i', filename,
                                  '-y',
                                  '-vn',
                                  '-ar', '44100',
                                  '-f', 'wav',
                                  temp_audiofile.name],
                                 stdin=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL,
                                 shell=shell)
            # Run FFMPEG in a subprocess and wait to finish
            p.communicate()
            # Load the temporary .wav-file
            audio_data, sfreq = librosa.load(temp_audiofile.name,
                                             sr=None,
                                             mono=False)
            # Delete the temporary file
            os.unlink(temp_audiofile.name)
        except Exception as ex:
            # If something went wrong then delete the temporary file if it
            # exists
            try:
                os.unlink(temp_audiofile.name)
            except Exception as ex:
                pass
            # Raise the original exception
            raise ex
    elif filename.lower().endswith(".wav"):
        # .wav files can be loaded directy with librosa
        audio_data, sfreq = librosa.load(filename, sr=None, mono=False)
    else:
        raise RuntimeError("Unknown input file format!")

    # Mono soundfiles come in a 1D array so turn them in to 2D
    if len(audio_data.shape) == 1:
        audio_data = np.array([audio_data])

    # Calculate the rest of the values based on data
    nof_channels = audio_data.shape[0]
    length_s = audio_data.shape[1]
    length_t = length_s / sfreq

    # Normalize the data to scale [-1, 1]
    maximum = np.max(np.abs(audio_data))
    audio_data = audio_data/maximum

    return audio_data, sfreq, nof_channels, length_s, length_t


def save_audio_data_to_wav(filename, audio_data, sfreq):
    """Save audio data from a NumPy array to a .wav file

    Args:
        filename(str): Name of the file
        audio_data(numpy.ndarray): The audio data in a NumPy array
        sfreq(float): Sampling frequency in Hz

    """

    # Normalize the data to scale [-2**15, 2**15] (16 bit)
    maximum = np.max(np.abs(audio_data))
    audio_data = (2**15 - 1) * audio_data/maximum
    converted = []

    # Convert channels to numpy.int16
    for channel in audio_data:
        converted.append(channel.astype(np.int16))

    # Transpose to NumPy recognised audio array and write
    converted = np.array(converted).T
    wavfile.write(filename, sfreq, converted)


def save_audio_data_to_mp3(filename, audio_data, sfreq, bitrate=320):
    """Save audio data from a NumPy array to a .mp3 file

    Args:
        filename(str): Name of the file
        audio_data(numpy.ndarray): The audio data in a NumPy array
        sfreq(float): Sampling frequency in Hz
        bitrate(int): Bitrate in kbps
    """

    nof_channels = audio_data.shape[0]

    # Normalize the data to scale [-2**15, 2**15] (16 bit data)
    maximum = np.max(np.abs(audio_data))
    if maximum != 0.0:
        audio_data = (2**15 - 1) * audio_data/maximum
    channels = audio_data.shape[0]

    # Convert to 16 numpy.int16
    converted = []
    for channel in audio_data:
        converted.append(channel.astype(np.int16))
    converted = np.array(converted).T.flatten()

    # Encode with Lame
    encoder = lameenc.Encoder()
    encoder.silence()  # Silence the stdout and stderr
    encoder.set_bit_rate(bitrate)
    encoder.set_in_sample_rate(int(sfreq))
    encoder.set_channels(nof_channels)
    encoder.set_quality(2)  # The highest quality
    mp3_data = encoder.encode(converted)

    # Flush encoded data and write
    mp3_data += encoder.flush()
    with open(filename, "wb") as mp3_file:
        mp3_file.write(mp3_data)


def save_audio_data(filename, audio_data, sfreq):
    """Save audio data from a NumPy array to an audio file

    .. note:: Supports .wav and .mp3 files at the moment

    Args:
        filename(str): Name of the file
        audio_data(numpy.ndarray): The audio data in a NumPy array
        sfreq(float): Sampling frequency in Hz
    """

    # FIXME: FFMPEG has more file types supported but they are not tested here
    if filename.lower().endswith(".wav"):
        save_audio_data_to_wav(filename, audio_data, sfreq)
    elif filename.lower().endswith(".mp3"):
        save_audio_data_to_mp3(filename, audio_data, sfreq)
    else:
        raise RuntimeError("Unknown output file format!")


def calc_audio_rfft(audio_data, sfreq):
    """Calculate the SciPy real-valued FFT for a PCM audio data

    Args:
        audio_data(numpy.ndarray): Audio data in a NumPy array
        sfreq(float): Sampling frequency in Hz

    Returns:
        (tuple): tuple containing:
            rfft_data(numpy.ndarray): Single sided FFT data
            freq_scale(numpy.ndarray): Frequencies for each data point in FFT
    """

    rfft_data = []
    for channel in audio_data:
        rfft_data.append(rfft(channel))
    rfft_data = np.array(rfft_data)
    sample_length = len(rfft_data[0])
    fres = sfreq / sample_length / 2
    freq_scale = np.linspace(0, fres * (sample_length - 1), sample_length)
    return rfft_data, freq_scale


def calc_audio_irfft(fft_data):
    """Calculate the PCM audio data from a SciPy single-sided FFT

    Args:
        audio_data(numpy.ndarray): Audio data in a NumPy array
        sfreq(float): Sampling frequency in Hz

    Returns:
        (tuple): tuple containing:
            rfft_data(numpy.ndarray): Single sided FFT data
            freq_scale(numpy.ndarray): Frequencies for each data point in FFT
    """
    audio_data = []
    for channel in fft_data:
        audio_data.append(irfft(channel))
    audio_data = np.array(audio_data)
    return audio_data


def get_channel_decomposition(channel, decompose):
    """Get a channel decomposition using median-filtering HPSS

    Args:
        channel(numpy.ndarray): Channel data in a NumPy array
        decompose(str): Decomposition type: "harmonic" or "percussive"

    Returns:
        (tuple): tuple containing
            new_channel(numpy.ndarray): The matching decomposed channel
            leftover(numpy.ndarray): The non-matching decomposed channel
    """
    harmonic, percussive = hpss(channel)
    if decompose == "harmonic":
        new_channel = harmonic
        leftover = percussive
    elif decompose == "percussive":
        new_channel = percussive
        leftover = harmonic

    return new_channel, leftover


def get_decomposed_audio(audio_data, decompose):
    """Get a audio decomposition using median-filtering HPSS

    Args:
        audio_data(numpy.ndarray): Audio data in a NumPy array
        decompose(str): Decomposition type: "harmonic" or "percussive"

    Returns:
        (tuple): tuple containing
            new_audio_data(numpy.ndarray): The matching decomposed audio
            leftover(numpy.ndarray): The non-matching decomposed audio

    Raises:
        RuntimeError
    """

    if decompose in ["percussive", "harmonic"]:
        new_audio_data = []
        leftover = []
        for channel in audio_data:
            nc, lo = get_channel_decomposition(channel, decompose)
            new_audio_data.append(nc)
            leftover.append(lo)
        new_audio_data = np.array(new_audio_data)
        leftover = np.array(leftover)
    elif decompose == "none":
        new_audio_data = audio_data
        leftover = None
    else:
        raise RuntimeError(f"Unrecognized decomposition type: {decompose}")

    return new_audio_data, leftover


def deharmonize(audio_data, sfreq, shift, high=False,
                audio_min_freq=200.0, decompose="none"):
    """Deharmonize audio data using full signal FFT

    Args:
        audio_data(numpy.ndarray): Audio data in a NumPy array
        sfreq(float): Sampling frequency in Hz
        shift(float): Linear shift in Hz
        high(bool): True if shifting towards high frequencies
        audio_min_freq(float): Low cut off of shift in Hz
        decompose(str): Decomposition type:
            "none" -- No decomposition
            "harmonic" -- Perform operation only on harmonic components
            "percussive" -- Perform operation only on percussive components

    Returns:
        (numpy.ndarray): The deharmonized audio in a NumPy array
    """
    # Do the decomposition (if requested)
    audio_data, leftover = get_decomposed_audio(audio_data, decompose)

    # Calculate single-sided FFT
    fft_data, freq_scale = calc_audio_rfft(audio_data, sfreq)

    # Find the index of the low cut off frequency
    audio_cut_index = np.where(freq_scale >= audio_min_freq)[0][0]

    if not high:
        # Linear shift to low frequencies

        # The upper limit of the cut
        freq_cut_index = np.where(freq_scale >= (audio_min_freq + shift))[0][0]

        # Zero padding to keep the signal the same length
        padding = np.zeros(freq_cut_index - audio_cut_index)
        d_fft = []

        # Cut and append zero-padding to each channel
        for channel in fft_data:
            d_fft.append(np.concatenate([channel[:audio_cut_index],
                                        channel[freq_cut_index:],
                                        padding]))
    else:
        # Linear shift to high frequencies

        # The upper limit of the cut and its index
        max_freq = np.max(freq_scale)
        freq_cutoff = max_freq - shift
        freq_cut_index = np.where(freq_scale >= (freq_cutoff))[0][0]

        # Zero padding to keep the signal the same length
        padding = np.zeros(len(freq_scale) - freq_cut_index)
        d_fft = []

        # Cut and append zero-padding to each channel
        for channel in fft_data:
            d_fft.append(np.concatenate([channel[:audio_cut_index],
                                        padding,
                                        channel[audio_cut_index:
                                                freq_cut_index]]))
    d_fft = np.array(d_fft)

    # Mix back the decomposed left-over part
    if decompose in ["percussive", "harmonic"]:
        decomposed, fs = calc_audio_rfft(leftover, sfreq)
        d_fft += decomposed

    # Transform back to time-domain
    deharmonized_audio = calc_audio_irfft(d_fft)
    return deharmonized_audio


def deharmonize_slots_low(stft_datas, freq_cut_index, audio_cut_index,
                          padding):
    """Deharmonize a set of short-time FFT slots to lower frequencies

    Args:
        stft_datas(list): A list of NumPy arrays containint STFT blocks
        freq_cut_index(int): The array index of upper limit of cut
        audio_cut_index(int): The array index of audio low cut off
        padding(numpy.ndarray): The zero padding to be used

    Returns:
        (list): List of NumPy arrays containing the deharmonized STFT blocks
    """

    istft_datas = []
    for channel in stft_datas:
        channel_datas = []
        # Cut and add zero padding to each channel and add to return array
        for time_slot in channel:
            new_slot = np.concatenate([time_slot[:audio_cut_index],
                                       time_slot[freq_cut_index:],
                                       padding])
            channel_datas.append(new_slot)
        istft_datas.append(channel_datas)
    return istft_datas


def deharmonize_slots_high(stft_datas, freq_cut_index, audio_cut_index,
                           padding):
    """Deharmonize a set of short-time FFT slots to higher frequencies

    Args:
        stft_datas(list): A list of NumPy arrays containint STFT blocks
        freq_cut_index(int): The array index of upper limit of cut
        audio_cut_index(int): The array index of audio low cut off
        padding(numpy.ndarray): The zero padding to be used

    Returns:
        (list): List of NumPy arrays containing the deharmonized STFT blocks
    """

    istft_datas = []
    for channel in stft_datas:
        channel_datas = []
        # Cut and add zero padding to each channel and add to return array
        for time_slot in channel:
            new_slot = np.concatenate([time_slot[:audio_cut_index],
                                       padding,
                                       time_slot[audio_cut_index:
                                                 freq_cut_index + 1]])
            channel_datas.append(new_slot)
        istft_datas.append(channel_datas)
    return istft_datas


def deharmonize_stft(audio_data, sfreq, shift, high=False,
                     audio_min_freq=200.0, decompose="none", nperseg=512):
    """Deharmonize audio data using short-time FFT

    Args:
        audio_data(numpy.ndarray): Audio data in a NumPy array
        sfreq(float): Sampling frequency in Hz
        shift(float): Linear shift in Hz
        high(bool): True if shifting towards high frequencies
        audio_min_freq(float): Low cut off of shift in Hz
        decompose(str): Decomposition type:
            "none" -- No decomposition
            "harmonic" -- Perform operation only on harmonic components
            "percussive" -- Perform operation only on percussive components
        nperseg(int): Samples per FFT block

    Returns:
        (numpy.ndarray): The deharmonized audio in a NumPy array
    """

    orig_length = audio_data.shape[1]

    # Decompose audio (if requested)
    audio_data, leftover = get_decomposed_audio(audio_data, decompose)
    stft_datas = []

    # Calculate short-time FFT blocks for all channels
    for channel in audio_data:
        sf, time_segments, stft_data = stft(channel, nperseg=nperseg,
                                            noverlap=0.75 * nperseg,
                                            padded=False)
        stft_data = stft_data.T
        sample_length = len(stft_data[0])
        fres = sfreq / sample_length / 2
        freq_scale = np.linspace(0, fres * (sample_length - 1), sample_length)
        stft_datas.append(stft_data)

    # Get the index of the low cut off frequency
    audio_cut_index = np.where(freq_scale >= audio_min_freq)[0][0]
    istft_datas = []

    if not high:
        # Linear shift to low frequencies

        # Get the index of the cut frequency
        freq_cut_index = np.where(freq_scale >= (audio_min_freq + shift))[0][0]

        # Zero padding array
        padding = np.zeros(freq_cut_index - audio_cut_index)

        # Calculate the new short-time FFT blocks
        istft_datas = deharmonize_slots_low(stft_datas,
                                            freq_cut_index,
                                            audio_cut_index,
                                            padding)
    else:
        # Linear shift to high frequencies

        # Get the index of the cut frequency
        max_freq = np.max(freq_scale)
        freq_cutoff = max_freq - shift
        freq_cut_index = np.where(freq_scale >= (freq_cutoff))[0][0]

        # Zero padding array
        padding = np.zeros(len(freq_scale) - freq_cut_index - 1)

        # Calculate the new short-time FFT blocks
        istft_datas = deharmonize_slots_high(stft_datas,
                                             freq_cut_index,
                                             audio_cut_index,
                                             padding)

    deharmonized_audio = []

    # Transform all channels back to time-domain
    for channel in istft_datas:
        channel = np.array(channel).T
        data_times, deharm = istft(channel, nperseg=nperseg,
                                    noverlap=0.75 * nperseg)
        # ISTFT doesn't necessarily produce exactly the same length signal
        # so resample to original length
        deharm = resample(deharm, orig_length)
        deharmonized_audio.append(deharm)

    deharmonized_audio = np.array(deharmonized_audio)

    # Mix back the left-over signal from decomposition if available
    if leftover is not None:
        deharmonized_audio += leftover
    return deharmonized_audio

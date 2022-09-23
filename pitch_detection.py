import copy
import numpy as np
import scipy.fftpack
import soundfile

# General settings that can be changed by the user
SAMPLE_FREQ = 16000  # sample frequency in Hz
WINDOW_SIZE = 16000  # window size of the DFT in samples
NUM_HPS = 5  # max number of harmonic product spectrums
CONCERT_PITCH = 440  # defining a1
ALL_NOTES = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]

HANN_WINDOW = np.hanning(WINDOW_SIZE)


def find_pitch(in_data):
    hann_samples = in_data * HANN_WINDOW
    magnitude_spec = abs(scipy.fftpack.fft(hann_samples)[:len(hann_samples) // 2])

    # interpolate spectrum
    mag_spec_ipol = np.interp(np.arange(0, len(magnitude_spec), 1 / NUM_HPS), np.arange(0, len(magnitude_spec)),
                              magnitude_spec)
    mag_spec_ipol = mag_spec_ipol / np.linalg.norm(mag_spec_ipol, ord=2)  # normalize it

    hps_spec = copy.deepcopy(mag_spec_ipol)

    for i in range(NUM_HPS):
        tmp_hps_spec = np.multiply(hps_spec[:int(np.ceil(len(mag_spec_ipol) / (i + 1)))], mag_spec_ipol[::(i + 1)])
        if not any(tmp_hps_spec):
            break
        hps_spec = tmp_hps_spec

    max_ind = np.argmax(hps_spec)
    max_freq = max_ind * (SAMPLE_FREQ / WINDOW_SIZE) / NUM_HPS

    i = int(np.round(np.log2(max_freq / CONCERT_PITCH) * 12))
    closest_note = ALL_NOTES[i % 12] + str(4 + (i + 9) // 12)
    closest_pitch = CONCERT_PITCH * 2 ** (i / 12)

    return max_freq, round(closest_pitch), closest_note


data, fs = soundfile.read('./test.wav')
pitch, closet_pitch, closest_note = find_pitch(data[:WINDOW_SIZE])

print('{}Hz'.format(pitch), '{}Hz'.format(closet_pitch), closest_note)

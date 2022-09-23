import copy
import numpy as np
import scipy.fftpack
import sounddevice as sd
import time

# General settings that can be changed by the user
SAMPLE_FREQ = 48000  # sample frequency in Hz
WINDOW_SIZE = 48000
WINDOW_STEP = 12000
POWER_THRESH = 1e-6
NUM_HPS = 5
HANN_WINDOW = np.hanning(WINDOW_SIZE)
CONCERT_PITCH = 440  # defining a4
ALL_NOTES = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]


def callback(indata, frames, time, status):
    """
    Callback function of the InputStream method.
    """
    # define static variables
    if not hasattr(callback, "window_samples"):
        callback.window_samples = [0 for _ in range(WINDOW_SIZE)]
    if not hasattr(callback, "noteBuffer"):
        callback.noteBuffer = ["1", "2"]

    if status:
        print(status)
        return
    if any(indata):
        callback.window_samples = np.concatenate((callback.window_samples, indata[:, 0]))  # append new samples
        callback.window_samples = callback.window_samples[len(indata[:, 0]):]  # remove old samples

        # skip if signal power is too low
        signal_power = (np.linalg.norm(callback.window_samples, ord=2) ** 2) / len(callback.window_samples)
        if signal_power < POWER_THRESH:
            print("입력없음")
            return
        else:
            # print("입력있음")
            hann_samples = callback.window_samples * HANN_WINDOW
            magnitude_spec = abs(scipy.fftpack.fft(hann_samples)[:len(hann_samples) // 2])

            # interpolate spectrum
            mag_spec_ipol = np.interp(np.arange(0, len(magnitude_spec), 1 / NUM_HPS), np.arange(0, len(magnitude_spec)),
                                      magnitude_spec)
            mag_spec_ipol = mag_spec_ipol / np.linalg.norm(mag_spec_ipol, ord=2)  # normalize it

            hps_spec = copy.deepcopy(mag_spec_ipol)

            for i in range(NUM_HPS):
                tmp_hps_spec = np.multiply(hps_spec[:int(np.ceil(len(mag_spec_ipol) / (i + 1)))],
                                           mag_spec_ipol[::(i + 1)])
                if not any(tmp_hps_spec):
                    break
                hps_spec = tmp_hps_spec

            max_ind = np.argmax(hps_spec)
            max_freq = max_ind * (SAMPLE_FREQ / WINDOW_SIZE) / NUM_HPS

            i = int(np.round(np.log2(max_freq / CONCERT_PITCH) * 12))
            closest_note = ALL_NOTES[i % 12] + str(4 + (i + 9) // 12)
            closest_pitch = CONCERT_PITCH * 2 ** (i / 12)

            print('{}Hz'.format(max_freq), '{}Hz'.format(round(closest_pitch)), closest_note)
    else:
        print('no input')


try:
    with sd.InputStream(channels=1, callback=callback, blocksize=WINDOW_STEP, samplerate=SAMPLE_FREQ):
        while True:
            time.sleep(0.5)
except Exception as exc:
    print(str(exc))

# -*- coding: utf-8 -*-
# main.py
# author : Antoine Passemiers

import os, sys
import numpy as np
from scipy.io.wavfile import read as scipy_read


CENTER_CHANNEL = -1

class MemoryManager(object):
    READING_ON_HARD_DISK = False
    WAV_HEADER_SIZE = 44

    def __init__(self, center_wav_filepath, filepaths):
        self.mmap = None
        self.filepaths = filepaths
        self.center_wav_filepath = center_wav_filepath
        self.cursors   = np.zeros(len(filepaths), dtype = np.int)

        self.setOutputWavPath("tmp.wav")

    def setOutputWavPath(self, filepath):
        self.outpath   = filepath
        self.open(CENTER_CHANNEL, raw = True)
        assert(MemoryManager.WAV_HEADER_SIZE % self.mmap.itemsize == 0)
        self.header_size = MemoryManager.WAV_HEADER_SIZE / self.mmap.itemsize
        self.header = np.copy(self.mmap[:self.header_size])
        self.outshape  = self.mmap.shape
        self.outdtype  = self.mmap.dtype
        self.write(self.header, 0)
        self.close()
    def getSignalFromFilepath(self, filepath):
        framerate, signal = scipy_read(filepath, mmap = True)
        assert(len(signal.shape) == 1)
        return signal
    def open(self, i, raw = False):
        if MemoryManager.READING_ON_HARD_DISK:
            self.close()
        filepath = self.center_wav_filepath if (i == CENTER_CHANNEL) else self.filepaths[i]
        if not raw:
            self.mmap = self.getSignalFromFilepath(filepath)
        else:
            self.mmap = np.memmap(filepath)
        MemoryManager.READING_ON_HARD_DISK = True
    def close(self):
        del self.mmap
        MemoryManager.READING_ON_HARD_DISK = False
    def write(self, signal, offset):
        if MemoryManager.READING_ON_HARD_DISK:
            self.close()
        self.mmap = np.memmap(
            self.outpath, 
            mode   = "r+",
            offset = offset,
            dtype  = self.outdtype,
            shape  = self.outshape)
        self.mmap[:len(signal)] = signal[:]
        self.mmap.flush()



def main():
    if len(sys.argv) < 3:
        print(
            """Error. Program must be called with at least 2 arguments :
            First argument  : path to the wav file of the center channel
            Second argument : path to the folder containing the other channels""")
        return
    center_wav_filepath = sys.argv[1]
    wav_folder          = sys.argv[2]
    filepaths = [os.path.join(wav_folder, filename) for filename in os.listdir(wav_folder)]

    """ EXAMPLE OF USE """
    manager = MemoryManager(center_wav_filepath, filepaths)
    manager.open(CENTER_CHANNEL) # Opens the center channel
    manager.open(0) # Closes the center channel and opens the L channel
    # manager.mmap contains the data samples of the current channel

    white_noise = np.random.rand(1000000) * 500
    manager.write(white_noise, 80) # Write a white noise in the output file
    # This is still raw data, a wave header must be written first


if __name__ == "__main__":
    main()
    print("Finished")
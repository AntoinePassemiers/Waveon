# -*- coding: utf-8 -*-
# main.py
# author : Antoine Passemiers

import os, sys
import numpy as np
from scipy.io.wavfile import read as scipy_read


class MemoryManager(object):
    READING_ON_HARD_DISK = False

    def __init__(self, filepaths):
        self.mmap = None
        self.filepaths = filepaths
        self.cursors   = np.zeros(len(filepaths), dtype = np.int)

        self.outpath   = "tmp"
        self.open(0)
        self.outshape  = self.mmap.shape
        self.outdtype  = self.mmap.dtype
        self.close()
    def setOutputWavPath(self, filepath):
        self.outpath = filepath
    def getSignalFromFilepath(self, filepath):
        framerate, signal = scipy_read(filepath, mmap = True)
        assert(len(signal.shape) == 1)
        return signal
    def open(self, i):
        if MemoryManager.READING_ON_HARD_DISK:
            self.close()
        self.mmap = self.getSignalFromFilepath(self.filepaths[i])
        MemoryManager.READING_ON_HARD_DISK = True
    def close(self):
        del self.mmap
        MemoryManager.READING_ON_HARD_DISK = False
    def write(self, signal, offset):
        if MemoryManager.READING_ON_HARD_DISK:
            self.close()
        self.mmap = np.memmap(
            self.outpath, 
            mode   = "w+",
            offset = offset,
            dtype  = self.outdtype,
            shape  = self.outshape)
        self.mmap[:len(signal)] = signal[:]


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
    manager = MemoryManager([center_wav_filepath] + filepaths)
    manager.open(0) # Opens the center channel
    manager.open(1) # Closes the center channel and opens the L channel
    # manager.mmap contains the data samples of the current channel

    manager.setOutputWavPath(os.path.join(wav_folder, "tmp"))
    manager.write(np.arange(45), 0) # Write 45 samples in the output
    # This is still raw data, a wave header must be written first


if __name__ == "__main__":
    main()
    print("Finished")
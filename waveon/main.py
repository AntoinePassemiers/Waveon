# -*- coding: utf-8 -*-
# main.py
# author : Antoine Passemiers

import os, sys
import numpy as np
from scipy.io.wavfile import read as scipy_read

from memory import *


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
    memory = MemoryManager(center_wav_filepath, filepaths)
    segment_size = Parameters.mmap_segment_size
    for i in range(len(memory) / segment_size):
        memory.open(CENTER_CHANNEL)
        segment = memory[i*segment_size:(i+1)*segment_size] # Load segment from center channel
        memory.open(0)
        left_channel = memory[i*segment_size:(i+1)*segment_size]
        segment -= 5 * left_channel
        memory.open(1)
        right_channel = memory[i*segment_size:(i+1)*segment_size]
        segment -= 5 * right_channel
        memory.write(segment, offset = i*segment_size) # Copy segment into the output file at position i*segment_size


if __name__ == "__main__":
    main()
    print("Finished")
# -*- coding: utf-8 -*-
# main.py - Memory management for handling large wav files
# author : Antoine Passemiers

import os, sys
import numpy as np
from scipy.io.wavfile import read as scipy_read


CENTER_CHANNEL = -1
OUTPUT_CHANNEL = -2

class Parameters:
    mmap_segment_size = 2 ** 22 # Number of audio samples per segment

class MemoryManager(object):
    READING_ON_DISK = False # True if a wav file is being accessed on the disk
    WAV_HEADER_SIZE = 44    # Number of bytes of a standard wave header

    """ Abstract object for loading, accessing and writing into large wav files.

    Parameters
    ----------
    center_wav_filepath: string
        Path to the wav file containing the center channel
    filepaths: list
        List of paths to the other wav files
        L channel, R channel, ...

    Attributes
    ----------
    mmap: np.memmap
        Memory map to a wav file located on the disk
        Only one memory map can be allocated at one time,
        this is why mmap only stores a reference to the current memmap
    outpath: string
        Path to the output wav file
    n_samples: int
        Number of audio samples if the center channel / output file
    itemsize: int
        Number of bytes per audio sample
    raw_itemsize: int
        Number of bytes per element in a raw file (supposed to be 1)
    outshape: tuple
        Shape of the center channel / output file
    outdtype: np.dtype
        Data type of the center channel / output file
    header: np.ndarray
        Array containing the header of the center channel / output file
    """
    def __init__(self, center_wav_filepath, filepaths):
        self.mmap = None
        self.filepaths = filepaths
        self.center_wav_filepath = center_wav_filepath
        self.setOutputWavPath("tmp.wav")

    def setOutputWavPath(self, filepath):
        """ Set the location of the output wav file 

        Parameters
        ----------
        filepath: string
            path to the output wav file
        """
        # Set the output path
        self.outpath   = filepath

        # Get the type, itemsize and number of audio samples
        self.open(CENTER_CHANNEL, raw = False)
        self.n_samples = self.mmap.shape[0]
        self.itemsize = self.mmap.itemsize
        self.outshape  = (self.n_samples,)
        self.outdtype  = self.mmap.dtype
        self.close()

        # Get the header of the center channel wavfile
        # This can only be achieved by accessing the wave file as a binary file
        self.open(CENTER_CHANNEL, raw = True)
        assert(MemoryManager.WAV_HEADER_SIZE % self.itemsize == 0)
        self.raw_itemsize = self.mmap.itemsize
        # Note the difference between self.itemsize and self.raw_itemsize
        header_size = MemoryManager.WAV_HEADER_SIZE / self.raw_itemsize
        self.header = np.copy(self.mmap[:header_size])
        self.close()

        # Create a binary wav file at the location pointed by filepath
        self.createOutputFile(self.header)
        self.close()

    def createOutputFile(self, header):
        """ Create a binary wave file from a raw header

        Parameters
        ----------
        header: np.ndarray
            Array containing a header in a binary format
        """
        # Create the wave file
        self.mmap = np.memmap(
            self.outpath, 
            mode   = "w+",
            offset = 0,
            dtype  = self.outdtype,
            shape  = self.outshape)
        self.close()

        # Load the same save file, but in binary format
        # Store the header at the beginning of it
        self.open(OUTPUT_CHANNEL, raw = True)
        self.mmap[:len(header)] = header[:]
        self.mmap.flush()

    def getSignalFromFilepath(self, filepath):
        """ Load a wave file as an audio array, without header,
        using scipy.io.wavfile

        Parameters
        ----------
        filepath: string
            Path to the wave file to open
        """
        if MemoryManager.READING_ON_DISK:
            self.close() # Deallocate the current memory map
        framerate, signal = scipy_read(filepath, mmap = True)
        assert(len(signal.shape) == 1) # Must be in mono format (1 channel)
        MemoryManager.READING_ON_DISK = True
        return signal

    def open(self, i, raw = False):
        """ Load a wave file using np.memmap, as a binary binary or an audio array
        
        Parameters
        ----------
        i: int
            Index of the wavefile / channel to open
            i must be in the range (0, len(self.filepaths)),
            or be equal to CENTER_CHANNEL or OUTPUT_CHANNEL
        raw: bool
            Whether to load the wavefile into a binary array or not
        """
        if MemoryManager.READING_ON_DISK:
            self.close() # Deallocate the current memory map
        if i == CENTER_CHANNEL:
            filepath = self.center_wav_filepath
        elif i == OUTPUT_CHANNEL:
            filepath = self.outpath
        else:
            filepath = self.filepaths[i]
        if not raw:
            # Load the file as an audio array (2 bytes per sample or more)
            self.mmap = self.getSignalFromFilepath(filepath)
        else:
            # Load the file as a binary array (1 byte per sample)
            self.mmap = np.memmap(filepath)
        MemoryManager.READING_ON_DISK = True

    def close(self):
        """ Deallocate the current memory map.
        Two files cannot be opened at the same time :
        This function must be called every time we want the memory map
        to point to a new file.
        """
        try:
            del self.mmap
        except AttributeError:
            pass # Do nothing if no memory map is allocated
        MemoryManager.READING_ON_DISK = False

    def write(self, signal, offset = 0):
        """ Open the output wave file and write a segment of data in it

        Parameters
        ----------
        signal: np.ndarray
            Segment to write into the output wave file
        offset: int
            Index of the audio sample where to start from.
            offset < self.__len__()
        """
        if MemoryManager.READING_ON_DISK:
            self.close()
        self.mmap = np.memmap(
            self.outpath, 
            mode   = "r+",
            offset = offset * self.itemsize + MemoryManager.WAV_HEADER_SIZE,
            dtype  = self.outdtype,
            shape  = self.outshape)
        self.mmap[:len(signal)] = signal[:]
        self.mmap.flush()
        MemoryManager.READING_ON_DISK = True

    def __getitem__(self, s):
        """ Access the audio samples located in s, using the current memory map

        Parameters
        ----------
        s: slice
            slice of array samples
        """
        assert(isinstance(s, slice))
        return np.copy(self.mmap[s])

    def __len__(self):
        """ Get the number of audio samples in the center channel / output file """
        return self.outshape[0]
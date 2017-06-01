"""This module contains data extraction functions for
Trodes (http://spikegadgets.com/software/trodes.html)

In development see issue #164 (https://github.com/eackermann/nelpy/issues/164)
"""

import warnings
import numpy as np
import os
from ..core import AnalogSignalArray


def load_lfp_dat(filepath, *,tetrode, channel, decimation_factor=-1):
    """Loads lfp and timestamps from .dat files into AnalogSignalArray after
    exportLFP function generates .LFP folder. This function assumes the names of
    the .LFP folder and within the .LFP folder have not been changed from defaults
    (i.e. they should be the same prior to tetrode and channel number and
    extentions). fs is automatically calculated from the .dat file info and 
    decimation factor provided. step size is also automatically calculated from 
    the extracted timestamps.

    Parameters
    ----------
    filepath : string
        filepath to .LFP file nothing further is required. See examples.
    tetrode : np.array(dtype=uint, dimension=N)
        Tetrode(s) to extract from. A singular tetrode can be listed more than once
        if more than one channel from that tetrode is requested. Size of tetrodes
        requested and size of channels requested must match.
    channel : np.array(dtype=uint, dimension=N)
        Channel(s) to extract data from. For each tetrode, given in the input the
        same number of channels must be given. See examples.
    decimate : int (optional)
        Factor by which data is decimated. Data will match what is sent to modules.
        This is initialized to -1 and not used by default. Intelligent decimation or
        interpolation is not done here. Load up AnalogSignalArray then do that if it
        is of importance.

    Returns
    ----------
    asa : AnalogSignalArray
        AnalogSignalArray containing timestamps and particular tetrode and channels
        requested

    Examples *need to be reworked after changes
    ----------
    >>> #Single channel (tetrode 1 channel 3) extraction with fs and step
    >>> load_lfp_dat("debugging/testMoo.LFP", 1, 3, fs=30000, step=10)
    out : AnalogSignalArray with given timestamps, fs, and step size

    >>> #Multichannel extraction with fs and step
    >>> #tetrode 1 channels 1 and 4, tetrodes 3, 6, and 8 channels 2, 1, and 3
    >>> load_lfp_dat("debugging/testMoo.LFP", [1,1,3,6,8],[1,4,2,1,3], fs=30000, step=10)
    out : AnalogSignalArray with given timestamps, fs, and step size

    """

    def get_fsacq(filePath):
        f = open(filePath,'rb')
        instr = f.readline()
        while (instr[0:11] != b'Clock rate:'):
            instr = f.readline()
        return float(str(instr[11:]).split(" ")[-1].split("\\n")[0])

    def load_timestamps(filePath, fs_acquisition):
        print("*****************Loading LFP Timestamps*****************")
        f = open(filePath,'rb')
        instr = f.readline()
        while (instr != b'<End settings>\n') :
            print(instr)
            instr = f.readline()
        print('Current file position', f.tell())
        timestamps = np.fromfile(f, dtype=np.uint32)
        print("Done")
        return timestamps/fs_acquisition

    def load_lfp(filePath):
        print("*****************Loading LFP Data*****************")
        f = open(filePath,'rb')
        instr = f.readline()
        while (instr != b'<End settings>\n') :
            print(instr)
            if(instr[0:16] == b'Voltage_scaling:'):
                voltage_scaling = np.float(instr[18:-1])
            instr = f.readline()
        print('Current file position', f.tell())
        data = np.fromfile(f, dtype=np.int16)*voltage_scaling
        print("Done")
        return data

    data = []
    #if .LFP file path was passed
    if(filepath[-4:len(filepath)] == ".LFP"):
        #get file name
        temp = filepath[0:-4].split('/')[-1]
        #store fs_acquisition
        fs_acquisition = get_fsacq(filepath + "/" + temp + ".timestamps.dat")
        #load up timestamp data
        timestamps = load_timestamps(filepath + "/" + temp + ".timestamps.dat",\
                                     fs_acquisition)
        step = np.mean(np.diff(timestamps))
        #if we're decimating start from the first index that's divisible by zero
        #this is done to match the data sent out to the trodes modules
        if(decimation_factor > 0):
            decimation_factor = np.int(decimation_factor)
            start = 0
            while(timestamps[start]%(decimation_factor*10) != 0):
                start+=1
            timestamps = timestamps[start::decimation_factor*10]
            #account for fs if it's decimated
            fs = fs_acquisition/decimation_factor*10
        else:
            #fs_acquisition should be the same as fs if there isn't decimation
            fs = fs_acquisition
        #load up lfp data
        tetrode = np.array(np.squeeze(tetrode),ndmin=1)
        channel = np.array(np.squeeze(channel),ndmin=1)
        if(len(tetrode) == len(channel)):
            for t in enumerate(tetrode):
                lfp = load_lfp(filepath + "/" + temp + ".LFP_nt" + str(t[1]) +\
                 "ch" + str(channel[t[0]]) + ".dat")
                if(decimation_factor > 0):
                    lfp = lfp[start::decimation_factor*10]
                data.append(lfp)
        else:
            raise TypeError("Tetrode and Channel dimensionality mismatch!")

        #make AnalogSignalArray
        asa = AnalogSignalArray(data, timestamps=timestamps, fs=fs, step=step)
    else:
        raise FileNotFoundError(".LFP extension expected")

    return asa

def load_dio_dat(filepath):
    """Loads DIO pin event timestamps from .dat files. Returns as 2D 
    numpy array containing timestamps and state changes aka high to low
    or low to high. NOTE: This will be changed to EventArray once it is
    implemented and has only been tested with digital input pins but it 
    should work with digital output pins because they are stored the 
    same way.

    Parameters
    ----------
    filepath : string
        Entire path to .dat file requested. See Examples. 

    Returns
    ----------
    events : np.array([uint32, uint8])
        numpy array of Trodes imestamps and state changes (0 or 1)
        First event is 0 or 1 (active high or low on pin) at first Trodes
        timestamp.

    Examples
    ----------
    >>> #Single channel (tetrode 1 channel 3) extraction with fs and step
    >>> load_dio_dat("twoChan_DONOTUSE.DIO/twoChan_DONOTUSE.dio_Din11.dat")
    out : numpy array of state changes [uint32 Trodes timestamps, uint8 0 or 1].

    """
    
    #DIO pin 11 is detection pulse
    print("*****************Loading DIO Data*****************")
    f = open(filepath,'rb')
    instr = f.readline()
    while (instr != b'<End settings>\n') :
        print(instr)
        instr = f.readline()
    print('Current file position', f.tell())
    #dt = np.dtype([np.uint32, np.uint8])
    #x = np.fromfile(f, dtype=dt)
    print("Done loading all data!")
    return np.asarray(np.fromfile(f, dtype=[('time',np.uint32), ('dio',np.uint8)]))

def load_dat(filepath):
    """Loads timestamps and unfiltered data from Trodes .dat files. These
    files are saved directly from Trodes. This function should _not_ be 
    used after exportLFP or exportDIO functions given in the Trodes repo
    have been run. This function is for loading .dat files that are saved
    instead of .rec files. This is generally done when the recording is 
    wireless and saved on an SD card. 
    """
    warnings.warn("This is not complete. Do NOT use.")
    raise DeprecationWarning("This should not fall under 'trodes', and is not much of a function yet")

    numChannels = 128
    headerSize = 10
    timestampSize = 4
    channelSize = numChannels*2
    packetSize = headerSize + timestampSize + channelSize

    timestamp = []
    chdata = []

    with open(filepath, 'rb') as fileobj:
        for packet in iter(lambda: fileobj.read(packetSize),''):
            ii += 1
            if packet:
                ts = struct.unpack('<I', packet[headerSize:headerSize+timestampSize])[0]
                timestamps.append(ts)
                ch = struct.unpack('<h', packet[headerSize+timestampSize:headerSize+timestampSize+2])[0]
                chdata.append(ch)
            else:
                break
            if ii > 1000000:
                break

def load_rec(filepath, trodesfilepath, *,tetrode, channel, userefs=False, \
             decimation_factor=-1):

    tetrode = np.array(np.squeeze(tetrode),ndmin=1)
    channel = np.array(np.squeeze(channel),ndmin=1)
    if (len(tetrode) != len(channel)):
        raise TypeError("Tetrode and Channel dimensionality mismatch!")
    channel_str = ','.join(str(x) for x in channel)
    tetrode_str = ','.join(str(x) for x in tetrode)

    os.system(trodesfilepath + "bin/exportLFP -rec " + '\"'+filepath+'\"' + \
              " -userefs " + '\"'+str(int(userefs))+'\"' + " -tetrode " + '\"' \
              +tetrode_str+'\"' + " -channel " + '\"'+channel_str+'\"')
    print(trodesfilepath + "bin/exportLFP -rec " + '\"'+filepath+'\"' + \
              " -userefs " + '\"'+str(int(userefs))+'\"' + " -tetrode " + '\"' \
              +tetrode_str+'\"' + " -channel " + '\"'+channel_str+'\"')
    return load_lfp_dat(filepath[:-4]+".LFP", tetrode=tetrode, channel=channel,\
                        decimation_factor = decimation_factor)
"""This module contains data extraction functions for
Trodes (http://spikegadgets.com/software/trodes.html)

In development see issue #164 (https://github.com/eackermann/nelpy/issues/164)
"""

import warnings
import numpy as np
import re
import os
import platform
from ..core import AnalogSignalArray


def load_tetrode_channel_nums(filepath, *, disable_tetrodes = None, \
                              disable_channels = None, verbose = False):
    """Loads up all tetrode and and channel numbers into numpy arrays. This is 
    primarily supposed to be an end-user helper function for specifying several 
    channels. This function, like Trodes tetrode and channel data and like 
    MATLAB because this is Python, is 1 indexed (i.e. there is *no tetrode X 
    channel 0-3; it is tetrode X channels 1-4)

    Parameters
    ----------
    filepath : string
        filepath to .rec file.
    disable_tetrodes : np.array(dtype=uint, dimension=N)
        Enter in tetrode(s) which are not desired. Default is None so all
        tetrodes will be loaded up. If disable_channels is left as None, all 
        channels from the tetrode will be discounted.
    disable_channel : np.array(dtype=uint, dimension=N)
        Enter in channel(s) which are not desired. Default is None so all 
        channels will be loaded up. If disable tetrodes is entered and this arg
        is set, dimensionality must match and specified channels of specified
        tetrodes will be disabled.

    Returns
    ----------
    tetrodes : np.array(dtype=uint, dimension=N)
        numpy array of all tetrodes that are requested and available based on 
        parameters above
    channels : np.array(dtype=uint, dimension=N)
        numpy array of all channels that are requested and available based on
        parameters above
    """    
    tetrodes = []
    channels = []
    #open the file!
    with open(filepath,'rb') as f:
        disable_tetrodes = np.asarray(disable_tetrodes)
        if(disable_channels is not None):
            disable_channels = np.asarray(disable_channels)
            if(disable_channels.shape != disable_tetrodes.shape):
                raise AttributeError("Dimensionality mismatch with disable_channels and disable_tetrodes")
        #read in line by line and check until we get to spikeconfig portion
        #all tetrodes used will be extracted from there
        instr = f.readline()
        spikeConfFound = False
        while (re.search(r'</Configuration>',str(instr)) is None):
            instr = f.readline()
            #check if we've made it to the spike config yet...
            if(not re.search(r'<SpikeConfiguration>',str(instr)) is None):
                spikeConfFound = True
            #if we're in the spike config portion let's extract tetrodes and
            #channels that are requested. 
            if(spikeConfFound):
                if(not re.search(r'</SpikeConfiguration>',str(instr)) is None):
                    break
                else:
                    #store tetrode and channel numbers that are requested.
                    if("id" in str(instr)):
                        #find tetrode number we're looking at
                        start_index = re.search(r'id=',str(instr)).end()+1
                        #check if id is multiple digits
                        end_index = start_index + 1
                        while(str(instr)[end_index] != '"'):
                            end_index += 1
                        tetrodenum = int(\
                                     str(instr)\
                                     [start_index:end_index])
                        #store all channels of tetrode if it's not disabled
                        #otherwise we'll skip the tetrode alltogether
                        if(not tetrodenum in disable_tetrodes):
                            for i in range(0,4):
                                tetrodes.append(tetrodenum)
                                channels.append(i+1)
                        #if particular channels of the tetrode are not wanted, 
                        #let's be nice and disable those, as requested
                        elif(disable_channels is not None):
                            chans = disable_channels[\
                                    np.where(disable_tetrodes == tetrodenum)]
                            if(verbose):
                                print("Disabling Tetrode {} | Channel(s) {}".\
                                      format(tetrodenum, chans))
                            for i in range(0,4):
                                if(not i+1 in chans):
                                    tetrodes.append(tetrodenum)
                                    channels.append(i+1)
                        else:
                            print("Disabling Tetrode {} ".format(tetrodenum))
        #handle strange case(s)...this should only pop up when you're trying to 
        #call a .rec file that isn't recording ephys data only DIOs.
        if(spikeConfFound == False):
            raise AttributeError("SpikeConfiguration not found in config of .rec")
        if tetrodes == [] or channels == []:
            warnings.warn("Tetrodes and channels arrays empty")
        if(verbose):
            print("Tetrodes: ",np.asarray(tetrodes))
            print("Channels: ", np.asarray(channels))
        return np.asarray(tetrodes), np.asarray(channels)

def load_digital_channel_nums(filepath, *, disable_digital_channels = None, \
                              verbose = False):
    """Loads up all digital input channels into a numpy arrays (will be changed
    to EventArray later). This is primarily supposed to be an end-user helper
    function for specifying several digital inputs channels. 

    Parameters
    ----------
    filepath : string
        filepath to .rec file.
    disable_digital_channels : np.array(dtype=uint, dimension=N)
        Enter in tetrode(s) which are not desired. Default is None so all
        tetrodes will be loaded up. If disable_channels is left as None, all 
        channels from the tetrode will be discounted.

    Returns
    ----------
    tetrodes : np.array(dtype=uint, dimension=N)
        numpy array of all tetrodes that are requested and available based on 
        parameters above
    channels : np.array(dtype=str, dimension=N)
        numpy array of all channels that are requested and available based on
        parameters above
    """    
    channels = []
    #open the file!
    with open(filepath,'rb') as f:
        disable_digital_channels = np.asarray(disable_digital_channels)
        #read in line by line and check until we get to spikeconfig portion
        #all tetrodes used will be extracted from there
        instr = f.readline()
        auxConfigFound = False
        while (instr != b'</Configuration>\n'):
            instr = f.readline()
            #check if we've made it to the spike config yet...
            if(instr == b' <AuxDisplayConfiguration>\n'):
                auxConfigFound = True
            #if we're in the auxiliary config portion let's extract digital
            #channels that are requested. 
            if(auxConfigFound):
                if(instr == b' </AuxDisplayConfiguration>\n'):
                    break
                else:
                    #store tetrode and channel numbers that are requested.
                    if("id" in str(instr)):
                        #find tetrode number we're looking at
                        if(re.search(r'id="Din',str(instr)) is not None):
                            digital_input_num = int(\
                                        str(instr)\
                                        [re.search(r'id="Din',str(instr)).end()])
                            #store all channels of tetrode if it's not disabled
                            #otherwise we'll skip the tetrode alltogether
                            if(not digital_input_num in disable_digital_channels):
                                channels.append(digital_input_num)
                                if(verbose):
                                    print("Channel in Array {}"\
                                                     .format(digital_input_num))
                            elif(verbose):
                                print("Channel Disabled {}".format(digital_input_num))
        #handle strange case(s)...this should only pop up when you're trying to 
        #call a .rec file that isn't recording ephys data only DIOs.
        if(auxConfigFound == False):
            raise AttributeError("Auxiliary Config not found in config of .rec")
        if channels == []:
            warnings.warn("digital inputs requested are empty")
        return np.asarray(channels)

def load_lfp_dat(filepath, *,tetrode, channel, decimation_factor=-1,\
                 trodes_style_decimation=False, verbose=False, labels=None):
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
    trodes_style_decimation : bool (optional)
        Decimation is done the same way as in Trodes with just taking every 10th
        sample as opposed to doing subsampling. By default this is set to False 
        which enables the usage of AnalogSignalArray's subsample function if 
        data is to be decimated. It is recommended to use subsample unless you 
        need the exact data that Trodes modules receive.
    labels : np.array(dtype=np.str,dimension=N)
        Labeling each one of the signals in ASA to be generated. By default this
        will be set to None. It is expected that all signals will be labeled if
        labels are passed in. If any signals are not labeled we will label them
        as Nones and if more labels are passed in than the number of signals
        given, the extras will be truncated. If we're nice (which we are for
        the most part), we will display a warning upon doing any of these
        things! :P Lastly, it is worth noting that most logical and type error
        checking for this is expected to be done by the user. Inputs are casted
        to strings and stored in a numpy array.

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
        """Extract acquisition fs from config portion of .dat file
        """
        with open(filePath, 'rb') as f:
            instr = f.readline()
            while (instr[0:11] != b'Clock rate:'):
                instr = f.readline()
        return float(str(instr[11:]).split(" ")[-1].split("\\n")[0])

    def load_timestamps(filePath, fs_acquisition):
        """Loads timestamps in units of time (seconds)
        """
        if(verbose):
            print("*****************Loading LFP Timestamps*****************")
        with open(filePath, 'rb') as f:
            instr = f.readline()
            while (instr != b'<End settings>\n') :
                if(verbose):
                    print(instr)
                instr = f.readline()
            if(verbose):
                print('Current file position', f.tell())
                print("Done")
            timestamps = np.fromfile(f, dtype=np.uint32)
        return timestamps/fs_acquisition

    def load_lfp(filePath):
        """Loads LFP data in uV.
        """
        if(verbose):
            print("*****************Loading LFP Data*****************")
        with open(filePath, 'rb') as f:
            instr = f.readline()
            while (instr != b'<End settings>\n') :
                if(verbose):
                    print(instr)
                if(instr[0:16] == b'Voltage_scaling:'):
                    voltage_scaling = np.float(instr[18:-1])
                instr = f.readline()
            if(verbose):
                print('Current file position', f.tell())
                print("Done")
            data = np.fromfile(f, dtype=np.int16)*voltage_scaling
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
        #if we want to do simple decimation (i.e. take every Xth sample)
        if(trodes_style_decimation and decimation_factor > 0):
        #if we're decimating start from the first index that's divisible by zero
        #this is done to match the data sent out to the trodes modules
            decimation_factor = np.int(decimation_factor)
            start = 0
            while(timestamps[start]%(decimation_factor*10) != 0):
                start+=1
            timestamps = timestamps[start::decimation_factor*10]
            #account for fs if it's decimated
            fs = fs_acquisition/(decimation_factor*10)
        else:
            #fs_acquisition should be the same as fs if there isn't decimation
            fs = fs_acquisition
        #appropriate step size after potential decimation
        step = np.mean(np.diff(timestamps))
        #load up lfp data
        tetrode = np.array(np.squeeze(tetrode),ndmin=1)
        channel = np.array(np.squeeze(channel),ndmin=1)
        if(len(tetrode) == len(channel)):
            for t in enumerate(tetrode):
                lfp = load_lfp(filepath + "/" + temp + ".LFP_nt" + str(t[1]) +\
                 "ch" + str(channel[t[0]]) + ".dat")
                if(decimation_factor > 0 and trodes_style_decimation):
                    lfp = lfp[start::decimation_factor*10]
                data.append(lfp)
        else:
            raise TypeError("Tetrode and Channel dimensionality mismatch!")

        #make AnalogSignalArray
        asa = AnalogSignalArray(data, timestamps=timestamps, fs=fs, step=step,\
                                labels=labels)
        #if we want a more robust decimation, let's subsample the ASA by the 
        #decimation factor
        if(decimation_factor > 0):
            decimation_factor = np.int(decimation_factor)
            asa = asa.subsample(fs=fs/(decimation_factor*10))
    else:
        raise FileNotFoundError(".LFP extension expected")

    return asa


def load_wideband_lfp_rec(filepath, trodesfilepath, *,tetrode, channel=None, userefs=False, \
             everything=False, decimation_factor=-1, trodes_style_decimation=False, \
             trodes_lowpass_filter_freq=-1, trodes_highpass_filter_freq=-1,\
             data_already_extracted=False, delete_files=False, verbose=False):
    """
    Loads wideband LFP from .rec file. See params and demo notebook.

    Parameters
    ----------
    filepath : string
        Entire filepath to .rec file (e.g. /home/kemerelab/Data/test.rec)
    trodesfilepath : string
        Filepath to trodes code directory (e.g. /home/kemerelab/Code/trodes/)
    tetrode : np.array(dtype=uint, dimension=N)
        Tetrode(s) to extract from. A singular tetrode can be listed more than once
        if more than one channel from that tetrode is requested. Size of tetrodes
        requested and size of channels requested must match.
    channel : np.array(dtype=uint, dimension=N)
        Channel(s) to extract data from. For each tetrode, given in the input the
        same number of channels must be given. See examples.
    userefs : bool (optional):
        Optional flag to enable reference subtraction based on what is specified 
        in the config file. By default this is set to False. It is recommended to
        remain False with no reference subtraction from the direct loading of the 
        data file into AnalogSignalArrays unless it is known that the config file 
        indeed has the right reference set and this isn't changed during the 
        recording session
    everything : bool (optional)
        Optional flag to load up all data from all tetrodes requested into
        AnalogSignalArrays. By default this is set to False.
    decimation_factor : uint (optional)
        Optional decimation factor to decimate the data. This will decimate the 
        data by piggy backing off AnalogSignalArray's subsample function unless 
        the trodes style decimation flag is set to true
    trodes_style_decimation : bool (optional)
        Decimation is done the same way as in Trodes with just taking every 10th
        sample as opposed to doing subsampling. By default this is set to False 
        which enables the usage of AnalogSignalArray's subsample function if 
        data is to be decimated. It is recommended to use subsample unless you 
        need the exact data that Trodes modules receive.
    trodes_lowpass_filter_freq : np.int()
        Flag to set lowpass filter frequency with Trodes inbuilt filters. This
        should only be used for real-time analysis as these are IIR filters and
        will cause a delay in the data. By default these filters are disabled.
    trodes_highpass_filter_freq : np.int()
        Flag to set highpass filter frequency with Trodes inbuilt filters. This
        should only be used for real-time analysis as these are IIR filters and
        will cause a delay in the data. By default these filters are disabled.
    data_already_extracted : bool (optional)
        This is a flag to stop the data from being extracted from a .rec to .dat
        files. By default we assume it has not been extracted but this can be
        set to True if it has and the function will work the same way.
    delete_files : bool (optional)
        This is a flag to delete the extracted lfp .dat files from the .rec. By
        default this is set to False and the files will not be deleted. Use at 
        your own discretion. 


    Returns
    ----------
    asa : list of AnalogSignalArrays or single AnalogSignalArray 
        All data requested from .rec file is loaded up into AnalogSignalArrays.
        It is worth noting that the returns are different based on what is
        requested. If specific tetrodes and channel numbers are requested they 
        are stored and labeled in a singular AnalogSignalArray; however, if 
        all channels are requested via the everything flag, all channels of a 
        tetrode are put into a single AnalogSignalArray and a list of 
        AnalogSignalArrays are returned with each AnalogSignalArray containing 
        4 channels of a tetrode.
    """
    tetrode = np.array(np.squeeze(tetrode),ndmin=1)
    #load all channels!
    if(everything):
        tetrode = np.unique(tetrode)
        if(not data_already_extracted):
            if(platform.system() == "Linux"):
                os.system(trodesfilepath + "bin/exportLFP -rec " + '\"'+filepath+'\"' + \
                        " -userefs " + '\"'+str(int(userefs))+'\"' + " -everything " + '\"' \
                        +"1"+"\"" +" -lowpass " + str(trodes_lowpass_filter_freq)\
                        +" -highpass " + str(trodes_highpass_filter_freq))
                if(verbose):
                    print(trodesfilepath + "bin/exportLFP -rec " + '\"'+filepath+'\"' + \
                            " -userefs " + '\"'+str(int(userefs))+'\"' + " -everything " + '\"' \
                            +"1"+"\""+" -lowpass " + str(trodes_lowpass_filter_freq)\
                        +" -highpass " + str(trodes_highpass_filter_freq))
            elif(platform.system() == "Windows"):
                os.system(trodesfilepath + "bin/win32/exportLFP.exe -rec " + '\"'+filepath+'\"' + \
                            " -userefs " + '\"'+str(int(userefs))+'\"' + " -lowpass " + str(trodes_lowpass_filter_freq)\
                            +" -highpass " + str(trodes_highpass_filter_freq) + " -everything " + '\"' \
                            +"1"+"\"")
                if(verbose):
                    print(trodesfilepath + "bin/win32/exportLFP.exe -rec " + '\"'+filepath+'\"' + \
                            " -userefs " + '\"'+str(int(userefs))+'\"' + " -lowpass " + str(trodes_lowpass_filter_freq)\
                            +" -highpass " + str(trodes_highpass_filter_freq) + " -everything " + '\"' \
                            +"1"+"\"")
            

        #return list of ASAs
        asa = []
        for i in range(len(tetrode)):
            
            #format labels
            tChars = np.chararray(4,) #4 channels per tetrode
            tChars[:] = 't'
            tChars = tChars.decode('UTF-8')
            cChars = np.chararray(4,)
            cChars[:] = 'c'
            cChars = cChars.decode('UTF-8')
            
            labels = np.core.defchararray.add(tChars, list(map(str, [tetrode[i]\
                                                           ,tetrode[i]\
                                                           ,tetrode[i]\
                                                           ,tetrode[i]])))
            labels = np.core.defchararray.add(labels, cChars)
            labels = np.core.defchararray.add(labels, list(map(str,[1,2,3,4])))

            asa.append(load_lfp_dat(filepath[:-4]+".LFP", tetrode= \
                                    [tetrode[i],tetrode[i],tetrode[i],\
                                    tetrode[i]], channel=[1,2,3,4], \
                                    decimation_factor = decimation_factor,\
                                    trodes_style_decimation = trodes_style_decimation,\
                                    labels = labels, verbose = verbose))
        if(delete_files and platform.system == "Linux"):
            # raise NotImplementedError("delete files not supported yet.")
            removeFile = filepath[:-3]+"LFP"
            os.system("rm -r " + removeFile)
        return asa

    #load specific channels
    else:
        if(channel == None):
            raise AttributeError("channels need to be specified if not extracting"\
                                 " everything aka all channels from tetrode X")
        channel = np.array(np.squeeze(channel),ndmin=1)
        if (len(tetrode) != len(channel)):
            raise TypeError("Tetrode and Channel dimensionality mismatch!")
        channel_str = ','.join(str(x) for x in channel)
        tetrode_str = ','.join(str(x) for x in tetrode)

        if(not data_already_extracted):
            if(platform.system() == "Linux"):
                os.system(trodesfilepath + "bin/exportLFP -rec " + '\"'+filepath+'\"' + \
                            " -userefs " + '\"'+str(int(userefs))+'\"' + " -tetrode " + '\"' \
                            +tetrode_str+'\"' + " -channel " + '\"'+channel_str+'\"'\
                            +" -lowpass " + str(trodes_lowpass_filter_freq)\
                            +" -highpass " + str(trodes_highpass_filter_freq))
                if(verbose):
                    print(trodesfilepath + "bin/exportLFP -rec " + '\"'+filepath+'\"' + \
                            " -userefs " + '\"'+str(int(userefs))+'\"' + " -tetrode " + '\"' \
                            +tetrode_str+'\"' + " -channel " + '\"'+channel_str+'\"'\
                            +" -lowpass " + str(trodes_lowpass_filter_freq)\
                        +" -highpass " + str(trodes_highpass_filter_freq))
            elif(platform.system() == "Windows"):
                os.system(trodesfilepath + "bin/win32/exportLFP.exe -rec " + '\"'+filepath+'\"' + \
                            " -userefs " + '\"'+str(int(userefs))+'\"' + " -lowpass " + str(trodes_lowpass_filter_freq)\
                            +" -highpass " + str(trodes_highpass_filter_freq) + " -tetrode " + '\"' \
                            +tetrode_str+'\"' + " -channel " + '\"'+channel_str+'\"'\
                            )
                if(verbose):
                    print(trodesfilepath + "bin/win32/exportLFP.exe -rec " + '\"'+filepath+'\"' + \
                            " -userefs " + '\"'+str(int(userefs))+'\"' + " -lowpass " + str(trodes_lowpass_filter_freq)\
                            +" -highpass " + str(trodes_highpass_filter_freq) + " -tetrode " + '\"' \
                            +tetrode_str+'\"' + " -channel " + '\"'+channel_str+'\"'\
                            )

        #format labels
        tChars = np.chararray(tetrode.shape)
        tChars[:] = 't'
        tChars = tChars.decode('UTF-8')
        cChars = np.chararray(channel.shape)
        cChars[:] = 'c'
        cChars = cChars.decode('UTF-8')
        
        labels = np.core.defchararray.add(tChars, list(map(str, tetrode)))
        labels = np.core.defchararray.add(labels, cChars)
        labels = np.core.defchararray.add(labels, list(map(str,channel)))

        #return ASA with requested data loaded
        asa = load_lfp_dat(filepath[:-4]+".LFP", tetrode=tetrode, channel=channel,\
                            decimation_factor = decimation_factor, \
                            verbose = verbose, labels = labels, \
                            trodes_style_decimation = trodes_style_decimation)
        if(delete_files and platform.system() == "Linux"):
            # raise NotImplementedError("delete files not supported yet.")
            removeFile = filepath[:-3]+"LFP"
            os.system("rm -r " + removeFile)
        return asa

def load_dio_dat(filepath, channel, verbose=False):
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
    if verbose:
        print("*****************Loading DIO Data*****************")
    
     #if .LFP file path was passed
    if(filepath[-4:len(filepath)] == ".DIO"):
        #get file name
        temp = filepath[0:-4].split('/')[-1]
        filepath = filepath + "/" + temp + ".dio_Din" + str(channel) + ".dat"

    else:
        raise FileNotFoundError(".DIO extension expected")

    with open(filepath, 'rb') as f:
        instr = f.readline()
        while (instr != b'<End settings>\n') :
            if verbose: 
                print(instr)
            instr = f.readline()
        if(verbose):
            print('Current file position', f.tell())
        returndata = np.asarray(np.fromfile(f, dtype=[('time',np.uint32), \
                                                      ('dio',np.uint8)]))
    #dt = np.dtype([np.uint32, np.uint8])
    #x = np.fromfile(f, dtype=dt)
    if(verbose):
        print("Done loading all data!")
    return returndata

def load_dio_rec(filepath, trodesfilepath, channel=None, *, delete_files=False,\
                 data_already_extracted=False, verbose=False):
    """<insert informative docstring here>
    """
    channel_str = ','.join("Din"+str(x) for x in channel)
    if(not data_already_extracted):
        if(platform.system() == "Linux"):
            os.system(trodesfilepath + "bin/exportdio -rec " + '\"'+filepath+'\"' + \
                    " -channel " + '\"'+channel_str+'\"')
            if(verbose):
                print(trodesfilepath + "bin/exportdio -rec " + '\"'+filepath+'\"' + \
                    " -channel " + '\"'+channel_str+'\"')
        elif(platform.system() == "Windows"):
            os.system(trodesfilepath + "bin/win32/exportdio.exe -rec " + '\"'+filepath+'\"' + \
                    " -channel " + '\"'+channel_str+'\"')
            if(verbose):
                os.system(trodesfilepath + "bin/win32/exportdio.exe -rec " + '\"'+filepath+'\"' + \
                    " -channel " + '\"'+channel_str+'\"')
    dios = []
    for i in range(len(channel)):
        datfilepath = filepath[:-3]+"DIO"
        if(verbose):
            dios.append(load_dio_dat(datfilepath, channel[i], verbose=True))
        else:
            dios.append(load_dio_dat(datfilepath, channel[i]))
    if(delete_files and platform.system() == "Linux"):
        removeFile = filepath[:-3]+"DIO"
        os.system("rm -r " + removeFile)
    return dios

def load_spike_dat(filepath, verbose=False):
    raise NotImplementedError("Yeah we don't support spikes yet...Anyways, Trodes spike detection doesn't really do much.")
    # Spike snippets with 40 points/snippet/channel.
    dt = np.dtype([('time', np.uint32), ('waveformCh1', np.int16, (40,)), 
                ('waveformCh2', np.int16, (40,)), ('waveformCh3', np.int16, (40,)),
                ('waveformCh4', np.int16, (40,))])

def load_spike_rec(filepath, trodesfilepath, *, delete_files=False,\
                 data_already_extracted=False, verbose=False):
    """
    """
    raise NotImplementedError("This function is under development but alternatives are provided in the examples.")
    if(not data_already_extracted):
        os.system(trodesfilepath + "bin/exportspikes -rec " + '\"'+filepath)
        if(verbose):
            print(trodesfilepath + "bin/exportspikes -rec " + '\"'+filepath)
    spikes = []
    for i in range(len(channel)):
        datfilepath = filepath[:-3]+"spikes"
        if(verbose):
            spikes.append(load_spike_dat(datfilepath, channel[i], verbose=True))
        else:
            spikes.append(load_spike_dat(datfilepath, channel[i]))
    if(delete_files):
        removeFile = filepath[:-3]+"spikes"
        os.system("rm -r " + removeFile)
    return spikes

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
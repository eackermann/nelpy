#include "lfpexporthandler.h"
#include <algorithm>
#include <iostream>

LFPExportHandler::LFPExportHandler(QStringList arguments):
    AbstractExportHandler(arguments)
{
    //Change defaults here
    outputSamplingRate = 30000;
    filterLowPass = -1;
    filterHighPass = -1;
    useSpikeFilters = false;
    useRefs = false;
    useOneChannelPerNTrode = false;

    parseArguments();


    //Parse custom arguments
    //parseCustomArguments(argumentsProcessed);

    if (argumentsProcessed != argumentList.length()-1) {
        _argumentsSupported = false;
        return;
    }
}

LFPExportHandler::~LFPExportHandler()
{

}

void LFPExportHandler::printHelpMenu() {
    //printf("-outputrate <integer>  -- The sampling rate of the output file. \n"
    //       );

    printf("\nUsed to extract LFP data from a raw rec file and save to individual files. \n");
    printf("Usage:  exportLFP -rec INPUTFILENAME OPTION1 VALUE1 OPTION2 VALUE2 ...  \n\n"
           "Input arguments \n");
    printf("Defaults:\n -outputrate 1500 \n -highpass 0 \n -lowpass 400 \n -interp -1 (inf) \n -userefs 0 \n");
    printf("-oneperntrode <0 or 1> -- if 1 (default) only the LFP channel is used for each ntrode. Otherwise all channels are exported.\n");
    //AbstractExportHandler::printHelpMenu();
}

void LFPExportHandler::parseArguments() {
    //Parse extra arguments not handled by the base class

    QString opn_string;
    int optionInd = 1;
    while (optionInd < argumentList.length()) {

        if ((argumentList.at(optionInd).compare("-h",Qt::CaseInsensitive)==0)) {
            //printCustomMenu();
            //return;
            printHelpMenu();
        } else if ((argumentList.at(optionInd).compare("-oneperntrode",Qt::CaseInsensitive)==0) && (argumentList.length() > optionInd+1)) {
            opn_string = argumentList.at(optionInd+1);
            optionInd++;
            argumentsProcessed = argumentsProcessed+2;
        }
        qDebug() << argumentList.at(optionInd);
        everything = 0;
        if(argumentList.at(optionInd).compare("-everything",Qt::CaseInsensitive)==0){
            argumentList.removeAt(optionInd);
            everything = argumentList.at(optionInd).toInt();
            argumentList.removeAt(optionInd);
        }
        if(argumentList.at(optionInd).compare("-tetrode",Qt::CaseInsensitive)==0 || argumentList.at(optionInd).compare("-tetrodes",Qt::CaseInsensitive)==0){
            argumentList.removeAt(optionInd);
            QRegExp rx("(\\ |\\,|\\.|\\:|\\t)"); //RegEx for ' ' or ',' or '.' or ':' or '\t'
            QStringList tetrodeNums = argumentList.at(optionInd).split(rx);
            for(int x = 0; x<tetrodeNums.size(); ++x){
                tetrodes.push_back(tetrodeNums[x].toInt()-1);
//                qDebug() << tetrodes[x];
            }
            argumentList.removeAt(optionInd);
        }
        if(argumentList.at(optionInd).compare("-channel",Qt::CaseInsensitive)==0 || argumentList.at(optionInd).compare("-channels",Qt::CaseInsensitive)==0 ){
            argumentList.removeAt(optionInd);
            QRegExp rx("(\\ |\\,|\\.|\\:|\\t)"); //RegEx for ' ' or ',' or '.' or ':' or '\t'
            QStringList tetrodeChans = argumentList.at(optionInd).split(rx);
            for(int x = 0; x<tetrodeChans.size(); ++x){
                channels.push_back(tetrodeChans[x].toInt()-1);
    //                qDebug() << channels[x];
            }
            argumentList.removeAt(optionInd);
        }
        optionInd++;
    }
    if(channels.size() != tetrodes.size()){
        qDebug() << "Each tetrode must have an associated channel for extraction!";
        AbstractExportHandler::printHelpMenu();
    }
    for(uint loopy=0; loopy<channels.size(); ++loopy){
        qDebug() << "Tetrode(s) requested | Channel Number" << tetrodes[loopy]+1 << "|" << channels[loopy]+1;
//        channelNumbers.push_back((tetrodes[loopy]*4)-(4-channels[loopy])-1);
//        qDebug() << "ChannelNumber" << channelNumbers[loopy];
    }
    if (!opn_string.isEmpty()) {
        bool ok1;

        useOneChannelPerNTrode = opn_string.toInt(&ok1);
        if (!ok1) {
            //Conversion to int didn't work
            qDebug() << "oneperchannel argument could not be resolved into an integer. Use 0 (false) or 1 (true)";
            argumentReadOk = false;
            return;
        }
    }

    AbstractExportHandler::parseArguments();
}

/*
void LFPExportHandler::printCustomMenu() {
    //printf("-outputrate <integer>  -- The sampling rate of the output file. \n"
    //       );

}*/

/*
void LFPExportHandler::parseCustomArguments(int &argumentsProcessed) {

    //Parse extra arguments not handled by the base class

    QString outputRate_string = "";

    int optionInd = 1;
    while (optionInd < argumentList.length()) {


        if ((argumentList.at(optionInd).compare("-h",Qt::CaseInsensitive)==0)) {
            printCustomMenu();
            return;
        }
        optionInd++;
    }

}*/

int LFPExportHandler::processData() {
    qDebug() << "Processing LFP...";

    //Calculate the packet positions for each channel that we are extracting, plus
    //other critical info (number of saved channels, reference info, etc).

    calculateChannelInfo();
    createFilters();


    if (!openInputFile()) {
        return -1;
    }


    //Create a directory for the output files located in the same place as the source file
    QFileInfo fi(recFileName);
    QString fileBaseName;
    if (outputFileName.isEmpty()) {
        fileBaseName = fi.baseName();
    } else {
        fileBaseName = outputFileName;
    }

    QString directory;
    if(outputDirectory.isEmpty()) {
        directory = fi.absolutePath();
    } else {
        directory = outputDirectory;
    }

    QString saveLocation = directory+QString(QDir::separator())+fileBaseName+".LFP"+QString(QDir::separator());
    QDir dir(saveLocation);
    if (!dir.exists()) {
        if (!dir.mkpath(".")) {
            qDebug() << "Error creating LFP directory.";
            return -1;
        }
    }

    QList<QFile*> neuroFilePtrs;
    QList<QDataStream*> neuroStreamPtrs;

    QList<QFile*> timeFilePtrs;
    QList<QDataStream*> timeStreamPtrs;

    //Create an output file for the timestamps
    //*****************************************
    timeFilePtrs.push_back(new QFile);
    timeFilePtrs.last()->setFileName(saveLocation+fileBaseName+QString(".timestamps.dat"));
    if (!timeFilePtrs.last()->open(QIODevice::WriteOnly)) {
        qDebug() << "Error creating output file.";
        return -1;
    }
    timeStreamPtrs.push_back(new QDataStream(timeFilePtrs.last()));
    timeStreamPtrs.last()->setByteOrder(QDataStream::LittleEndian);

    //Write the current settings to file
    QString infoLine;
    timeFilePtrs.last()->write("<Start settings>\n");
    infoLine = QString("Description: LFP timestamps\n");
    timeFilePtrs.last()->write(infoLine.toLocal8Bit());
    infoLine = QString("Byte_order: little endian\n");
    timeFilePtrs.last()->write(infoLine.toLocal8Bit());
    infoLine = QString("Original_file: ") + fi.fileName() + "\n";
    timeFilePtrs.last()->write(infoLine.toLocal8Bit());
    infoLine = QString("Clock rate: %1\n").arg(hardwareConf->sourceSamplingRate);
    timeFilePtrs.last()->write(infoLine.toLocal8Bit());
    infoLine = QString("Decimation: %1\n").arg(decimation);
    timeFilePtrs.last()->write(infoLine.toLocal8Bit());
    infoLine = QString("Time_offset: %1\n").arg(startOffsetTime);
    timeFilePtrs.last()->write(infoLine.toLocal8Bit());



    QString fieldLine;
    fieldLine.clear();
    fieldLine += "Fields: ";
    fieldLine += "<time uint32>";
    fieldLine += "\n";
    timeFilePtrs.last()->write(fieldLine.toLocal8Bit());
    timeFilePtrs.last()->write("<End settings>\n");
    timeFilePtrs.last()->flush();




    //Create an output file for the neural data
    //*****************************************




    //Create the output files (one for each channel) and write header sections
    for (int i=0; i<channelPacketLocations.length(); i++) {

        if ((useOneChannelPerNTrode && (nTrodeSettings[nTrodeIndForChannels[i]].lfpChannel == (channelInNTrode[i]))) || (!useOneChannelPerNTrode)) {

            neuroFilePtrs.push_back(new QFile);
            neuroFilePtrs.last()->setFileName(saveLocation+fileBaseName+QString(".LFP_nt%1ch%2.dat").arg(nTrodeForChannels[i]).arg(channelInNTrode[i]+1));
            if (!neuroFilePtrs.last()->open(QIODevice::WriteOnly)) {
                qDebug() << "Error creating output file.";
                return -1;
            }
            neuroStreamPtrs.push_back(new QDataStream(neuroFilePtrs.last()));
            neuroStreamPtrs.last()->setByteOrder(QDataStream::LittleEndian);

            //Write the current settings to file

            neuroFilePtrs.last()->write("<Start settings>\n");
            infoLine = QString("Description: LFP data for one channel\n");
            neuroFilePtrs.last()->write(infoLine.toLocal8Bit());
            infoLine = QString("Byte_order: little endian\n");
            neuroFilePtrs.last()->write(infoLine.toLocal8Bit());
            infoLine = QString("Original_file: ") + fi.fileName() + "\n";
            neuroFilePtrs.last()->write(infoLine.toLocal8Bit());
            infoLine = QString("nTrode_ID: %1\n").arg(nTrodeForChannels[i]);
            neuroFilePtrs.last()->write(infoLine.toLocal8Bit());
            infoLine = QString("nTrode_channel: %1\n").arg(channelInNTrode[i]+1);
            neuroFilePtrs.last()->write(infoLine.toLocal8Bit());
            infoLine = QString("Clock rate: %1\n").arg(hardwareConf->sourceSamplingRate);
            neuroFilePtrs.last()->write(infoLine.toLocal8Bit());
            infoLine = QString("Voltage_scaling: %1\n").arg(0.195);
            neuroFilePtrs.last()->write(infoLine.toLocal8Bit());
            infoLine = QString("Decimation: %1\n").arg(decimation);
            neuroFilePtrs.last()->write(infoLine.toLocal8Bit());
            infoLine = QString("First_timestamp: %1\n").arg(currentTimeStamp);
            neuroFilePtrs.last()->write(infoLine.toLocal8Bit());
            if (useRefs) {
                infoLine = QString("Reference: on\n");
            } else {
                infoLine = QString("Reference: off\n");
            }
            neuroFilePtrs.last()->write(infoLine.toLocal8Bit());
            if (filterLowPass > -1) {
                infoLine = QString("Low_pass_filter: %1\n").arg(filterLowPass);
            } else {
                infoLine = QString("Low_pass_filter: none\n");
            }
            neuroFilePtrs.last()->write(infoLine.toLocal8Bit());

            fieldLine.clear();
            fieldLine += "Fields: ";
            //fieldLine += "<time uint32>";
            fieldLine += "<voltage int16>";
            fieldLine += "\n";
            neuroFilePtrs.last()->write(fieldLine.toLocal8Bit());


            neuroFilePtrs.last()->write("<End settings>\n");
            neuroFilePtrs.last()->flush();
        }
    }

    if(everything == 0){
        for(int i = 0; i<channelPacketLocations.length(); ++i){
            bool keepIt = false;
            for(int x = 0; x<channels.size(); ++x){
                if(nTrodeIndForChannels[i] == tetrodes[x] && channelInNTrode[i] == channels[x]){
                    keepIt = true;
                    break;
                }
            }
            if(!keepIt){
                QFile removefile(saveLocation+fileBaseName+QString(".LFP_nt%1ch%2.dat").arg(nTrodeForChannels[i]).arg(channelInNTrode[i]+1));
                removefile.remove();
            }
        }
    }
    //************************************************
    int inputFileInd = 0;

    while (inputFileInd < recFileNameList.length()) {
        if (inputFileInd > 0) {
            //There are multiple files that need to be stiched together. It is assumed that they all have
            //the exact same header section.
            recFileName = recFileNameList.at(inputFileInd);
            uint32_t lastFileTStamp = currentTimeStamp;


            qDebug() << "\nAppending from file: " << recFileName;
            QFileInfo fi(recFileName);

            if (!fi.exists()) {
                qDebug() << "File could not be found: " << recFileName;
                break;
            }
            if (!openInputFile()) {
                qDebug() << "Error: it appears that the file does not have an identical header to the last file. Cannot append to file.";
                return -1;
            }
            for (int i=0; i < channelFilters.length(); i++) {
                channelFilters[i]->resetHistory();
            }
            if (currentTimeStamp < lastFileTStamp) {
                qDebug() << "Error: timestamps do not begin with greater value than the end of the last file. Aborting.";
                return -1;
            }



        }

        //Process the data and stream results to output files
        while(!filePtr->atEnd()) {
            //Read in a packet of data to make sure everything looks good
            if (!(filePtr->read(buffer.data(),filePacketSize) == filePacketSize)) {
                //We have reached the end of the file
                break;
            }
            //Find the time stamp
            bufferPtr = buffer.data()+packetTimeLocation;
            tPtr = (uint32_t *)(bufferPtr);
            currentTimeStamp = *tPtr+startOffsetTime;

            //The number of expected data points between this time stamp and the last (1 if no data is missing)
            int numberOfPointsToProcess = (currentTimeStamp-lastTimeStamp);
            if (numberOfPointsToProcess < 1) {
                qDebug() << "Warning: backwards timestamp at time " << currentTimeStamp << lastTimeStamp;
            }

            if ((maxGapSizeInterpolation > -1) && (numberOfPointsToProcess > (maxGapSizeInterpolation+1))) {
                //The gap in data is too large to interpolate over, so we reset the filters (maxGapSizeInterpolation of -1 means the no gap is too large)
                for (int i=0; i < channelFilters.length(); i++) {
                    channelFilters[i]->resetHistory();
                }
                numberOfPointsToProcess = 1;
                pointsSinceLastLog = -1;
            }

            if (numberOfPointsToProcess == 1) {
                //No missing data
                if (((pointsSinceLastLog+1)%decimation) == 0) {
                    *timeStreamPtrs.at(0) << currentTimeStamp;
                }
            } else {
                //Some missing data, which we will interpolate over
                for (int p=1; p<numberOfPointsToProcess+1; p++) {
                    if (((pointsSinceLastLog+p)%decimation) == 0) {
                        *timeStreamPtrs.at(0) << lastTimeStamp+p;
                    }
                }
            }
            int fileInd = 0;
//            qDebug() << channelPacketLocations.length();
            for (int chInd=0; chInd < channelPacketLocations.length(); chInd++) {
//                if(chInd == (2/*Tetrode 2*/*4)-(4-3/*Channel 3*/))
                if ((useOneChannelPerNTrode && (nTrodeSettings[nTrodeIndForChannels[chInd]].lfpChannel == (channelInNTrode[chInd]))) || (!useOneChannelPerNTrode)) {
                    bufferPtr = buffer.data()+channelPacketLocations[chInd];
                    tempDataPtr = (int16_t*)(bufferPtr);
                    tempDataPoint = *tempDataPtr;

                    if (useRefs && refOn[chInd]) {
                        //Subtract the digital reference
                        bufferPtr = buffer.data() + refPacketLocations[chInd];
                        tempDataPtr = (int16_t*)(bufferPtr);
                        tempRefPoint = *tempDataPtr;
                        tempDataPoint -= tempRefPoint;
                    }



                    if (numberOfPointsToProcess == 1) {
                        //No missing data

                        if (filterOn[chInd]) {
                            tempFilteredDataPoint = channelFilters.at(chInd)->addValue(tempDataPoint);
                        }

                        if (((pointsSinceLastLog+1)%decimation) == 0) {
                            if (filterOn[chInd]) {
                                if(everything == 0){
                                    for(int loopy=0; loopy<channels.size(); ++loopy){
                                        if(nTrodeIndForChannels[chInd] == tetrodes[loopy] && channelInNTrode[chInd] == channels[loopy])
                                            *neuroStreamPtrs.at(fileInd) << tempFilteredDataPoint; //1 file per channel
                                    }
                                }
                                else{
                                    *neuroStreamPtrs.at(fileInd) << tempFilteredDataPoint; //1 file per channel
                                }
                                //*neuroStreamPtrs.at(0) << tempFilteredDataPoint; //all channels in one file


                            } else {
                                if(everything == 0){
                                    for(int loopy=0; loopy<channels.size(); ++loopy){
                                        if(nTrodeIndForChannels[chInd] == tetrodes[loopy] && channelInNTrode[chInd] == channels[loopy])
                                            *neuroStreamPtrs.at(fileInd) << tempDataPoint; //1 file per channel
                                    }
                                }
                                else{
                                    *neuroStreamPtrs.at(fileInd) << tempDataPoint; //1 file per channel
                                }
                                //*neuroStreamPtrs.at(0) << tempFilteredDataPoint; //all channels in one file
                            }
                        }

                    } else {
                        //We have missing data, so we need to interpolate

                        int16_t interpolatedDataPtInt;
                        double interpolatedDataPtFloat;
                        for (int mp = 1; mp < numberOfPointsToProcess+1; mp++) {
                            interpolatedDataPtFloat = (double)dataLastTimePoint[chInd] + ((double)(mp)/numberOfPointsToProcess)*(tempDataPoint-dataLastTimePoint[chInd]);
                            interpolatedDataPtInt = round(interpolatedDataPtFloat);
                            if (filterOn[chInd]) {
                                interpolatedDataPtInt = channelFilters.at(chInd)->addValue(interpolatedDataPtInt);
                            }

                            if (((pointsSinceLastLog+mp)%decimation) == 0) {
                                if(everything == 0){
                                    for(int loopy=0; loopy<channels.size(); ++loopy){
                                        if(nTrodeIndForChannels[chInd] == tetrodes[loopy] && channelInNTrode[chInd] == channels[loopy])
                                            *neuroStreamPtrs.at(fileInd) << interpolatedDataPtInt; //1 file per channel
                                    }
                                }
                                else{
                                    *neuroStreamPtrs.at(fileInd) << interpolatedDataPtInt; //1 file per channel
                                }
                                //*neuroStreamPtrs.at(0) << interpolatedDataPtInt; //all channels in one file
                            }

                        }

                    }

                    dataLastTimePoint[chInd] = tempDataPoint; //remember the last unfilted data point
                    fileInd++;
                }
            }



            //Print the progress to stdout
            printProgress();

            lastTimeStamp = currentTimeStamp;
            pointsSinceLastLog = (pointsSinceLastLog+numberOfPointsToProcess)%decimation;


        }

        printf("\rDone\n");
        filePtr->close();
        inputFileInd++;

    }
//    qDebug() << neuroFilePtrs.length();
    for (int i=0; i < neuroFilePtrs.length(); i++) {
//        if(i == (2/*Tetrode 2*/*4)-(4-3/*Channel 3*/)){
        neuroFilePtrs[i]->flush();
//        }
        neuroFilePtrs[i]->close();
    }

    for (int i=0; i < timeFilePtrs.length(); i++) {
        timeFilePtrs[i]->flush();
        timeFilePtrs[i]->close();
    }

    return 0; //success
}

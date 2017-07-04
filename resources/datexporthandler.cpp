#include "datexporthandler.h"
#include <algorithm>
#include <iostream>

DATExportHandler::DATExportHandler(QStringList arguments):
    AbstractExportHandler(arguments)
{

    maxGapSizeInterpolation = 0;
    invertSpikes = true;

    parseArguments();

    /*
    Do custom argument checks here like this:
    if (ARGVAL != REQUIREMENT) {
        qDebug() << "Error: ....";
        argumentReadOk = false;
    }   
    */

    if ((maxGapSizeInterpolation > 0) || (maxGapSizeInterpolation < 0)) {
         qDebug() << "Error: interp must be 0.";
         argumentReadOk = false;
    }

    if (outputSamplingRate != -1) {
        qDebug() << "Error: outputrate must stay unchanged for spike processing.";
        qDebug() << outputSamplingRate;
        argumentReadOk = false;
    }

    //Parse custom arguments
    //parseCustomArguments(argumentsProcessed);

    if (argumentsProcessed != argumentList.length()-1) {
        _argumentsSupported = false;
        return;
    }
}

DATExportHandler::~DATExportHandler()
{

}

void DATExportHandler::printHelpMenu() {
    //printf("-outputrate <integer>  -- The sampling rate of the output file. \n"
    //       );

    printf("\nUsed to extract digital I/O channels from a raw rec file and save to individual files for each channel. \n");
    printf("Usage:  exportdio -rec INPUTFILENAME OPTION1 VALUE1 OPTION2 VALUE2 ...  \n\n"
           "Input arguments \n");
    //printf("Defaults:\n -invert 1 -outputrate -1 (full sampling, can't be changed') \n -usespikefilters 1 \n -interp 0 \n -userefs 1 \n\n\n");
    //printf("-invert <1 or 0> -- Whether or not to invert spikes to go upward\n");


    //AbstractExportHandler::printHelpMenu();
}

void DATExportHandler::parseArguments() {
    //Parse extra arguments not handled by the base class


    int optionInd = 1;
    while (optionInd < argumentList.length()) {

        if ((argumentList.at(optionInd).compare("-h",Qt::CaseInsensitive)==0)) {
            //printCustomMenu();
            //return;
            printHelpMenu();
        }
        else if(argumentList.at(optionInd).compare("-channel",Qt::CaseInsensitive)==0 || argumentList.at(optionInd).compare("-tetrodes",Qt::CaseInsensitive)==0){
            argumentList.removeAt(optionInd);
            QRegExp rx("(\\ |\\,|\\.|\\:|\\t)"); //RegEx for ' ' or ',' or '.' or ':' or '\t'
            channelId = argumentList.at(optionInd).split(rx);
            qDebug() << "Din Channel(s) Requested: ";
            for(int x = 0; x<channelId.size(); ++x){
                qDebug() << channelId[x];
            }
            argumentList.removeAt(optionInd);
        }
        optionInd++;
    }

    AbstractExportHandler::parseArguments();
}


int DATExportHandler::processData() {


    qDebug() << "Exporting DIO data...";

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

    QString saveLocation = directory+QString(QDir::separator())+fileBaseName+".DIO"+QString(QDir::separator());
    QDir dir(saveLocation);
    if (!dir.exists()) {
        if (!dir.mkpath(".")) {
            qDebug() << "Error creating DIO directory.";
            return -1;
        }
    }

    //Pointers to the neuro files
    /*QList<QFile*> neuroFilePtrs;
    QList<QDataStream*> neuroStreamPtrs;

    QList<QFile*> timeFilePtrs;
    QList<QDataStream*> timeStreamPtrs;*/



    QString infoLine;
    QString fieldLine;



    //Create an output file for the DIO data
    //*****************************************
    QList<QFile*> DIOFilePtrs;
    QList<QDataStream*> DIOStreamPtrs;
    QList<int> DIOChannelInds;
    QList<uint8_t> DIOLastValue;

    for (int auxChInd = 0; auxChInd < headerConf->headerChannels.length(); auxChInd++) {
        if (headerConf->headerChannels[auxChInd].dataType == DeviceChannel::DIGITALTYPE) {
            //only make and write to files of channels that are requested
            for(int ii = 0; ii < channelId.size(); ++ii){
                if(headerConf->headerChannels[auxChInd].idString  == channelId[ii]){
                    DIOChannelInds.push_back(auxChInd);
                    DIOFilePtrs.push_back(new QFile);
                    DIOLastValue.push_back(2); //state will be either 1 or 0, 2 means that it's the first time point
                    DIOFilePtrs.last()->setFileName(saveLocation+fileBaseName+QString(".dio_%1.dat").arg(headerConf->headerChannels[auxChInd].idString));
                    if (!DIOFilePtrs.last()->open(QIODevice::WriteOnly)) {
                        qDebug() << "Error creating output file.";
                        return -1;
                    }
                    DIOStreamPtrs.push_back(new QDataStream(DIOFilePtrs.last()));
                    DIOStreamPtrs.last()->setByteOrder(QDataStream::LittleEndian);

                    //Write the current settings to file

                    DIOFilePtrs.last()->write("<Start settings>\n");
                    infoLine = QString("Description: State change data for one digital channel. Display_order is 1-based\n");

                    DIOFilePtrs.last()->write(infoLine.toLocal8Bit());
                    infoLine = QString("Byte_order: little endian\n");
                    DIOFilePtrs.last()->write(infoLine.toLocal8Bit());
                    infoLine = QString("Original_file: ") + fi.fileName() + "\n";
                    DIOFilePtrs.last()->write(infoLine.toLocal8Bit());
                    if (headerConf->headerChannels[auxChInd].input) {
                        infoLine = QString("Direction: input\n");
                    } else {
                        infoLine = QString("Direction: output\n");
                    }
                    DIOFilePtrs.last()->write(infoLine.toLocal8Bit());
                    infoLine = QString("ID: %1\n").arg(headerConf->headerChannels[auxChInd].idString);
                    DIOFilePtrs.last()->write(infoLine.toLocal8Bit());
                    infoLine = QString("Display_order: %1\n").arg(auxChInd+1);
                    DIOFilePtrs.last()->write(infoLine.toLocal8Bit());
                    infoLine = QString("Clockrate: %1\n").arg(hardwareConf->sourceSamplingRate);
                    DIOFilePtrs.last()->write(infoLine.toLocal8Bit());
                    infoLine = QString("First_timestamp: %1\n").arg(currentTimeStamp);
                    DIOFilePtrs.last()->write(infoLine.toLocal8Bit());

                    fieldLine.clear();
                    fieldLine += "Fields: ";
                    fieldLine += "<time uint32>";
                    fieldLine += "<state uint8>";


                    fieldLine += "\n";
                    DIOFilePtrs.last()->write(fieldLine.toLocal8Bit());


                    DIOFilePtrs.last()->write("<End settings>\n");
                    DIOFilePtrs.last()->flush();
                }
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

        uint8_t tmpVal;
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
            currentTimeStamp = *tPtr + startOffsetTime;

            //The number of expected data points between this time stamp and the last (1 if no data is missing)
            int numberOfPointsToProcess = (currentTimeStamp-lastTimeStamp);
            if (numberOfPointsToProcess < 1) {
                qDebug() << "Warning: backwards timestamp at time " << currentTimeStamp << lastTimeStamp;
            }


            for (int chInd=0; chInd < DIOChannelInds.length(); chInd++) {

                bufferPtr = buffer.data()+headerConf->headerChannels[DIOChannelInds[chInd]].startByte;
                tmpVal = (uint8_t)((*bufferPtr & (1 << headerConf->headerChannels[DIOChannelInds[chInd]].digitalBit)) >> headerConf->headerChannels[DIOChannelInds[chInd]].digitalBit);

                if (tmpVal != DIOLastValue[chInd]) {
                    *DIOStreamPtrs.at(chInd) << currentTimeStamp << tmpVal;
                    DIOLastValue[chInd] = tmpVal;
                }
                /*
                bufferPtr = buffer.data()+channelPacketLocations[chInd];
                headerConf->headerChannels[auxChInd].startByte
                tempDataPtr = (int16_t*)(bufferPtr);
                tempDataPoint = *tempDataPtr;

                if (tempDataPoint == -32768) {
                    //This is a NAN, replace with 0
                    tempDataPoint = 0;
                } else if (useRefs && refOn[chInd]) {
                    //Subtract the digital reference
                    bufferPtr = buffer.data() + refPacketLocations[chInd];
                    tempDataPtr = (int16_t*)(bufferPtr);
                    tempRefPoint = *tempDataPtr;
                    tempDataPoint -= tempRefPoint;
                }

                dataLastTimePoint[chInd] = tempDataPoint; //remember the last unfiltered data point
                */
            }

            //Print the progress to stdout
            printProgress();

            lastTimeStamp = currentTimeStamp;
            //pointsSinceLastLog = (pointsSinceLastLog+numberOfPointsToProcess)%decimation;


        }
        filePtr->close();
        printf("\rDone\n");
        inputFileInd++;
    }



    for (int i=0; i < DIOFilePtrs.length(); i++) {
        DIOFilePtrs[i]->flush();
        DIOFilePtrs[i]->close();
    }

    return 0; //success
}


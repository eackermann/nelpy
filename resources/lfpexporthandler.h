#ifndef LFPEXPORTHANDLER_H
#define LFPEXPORTHANDLER_H
#include "abstractexporthandler.h"
#include "iirFilter.h"


class LFPExportHandler: public AbstractExportHandler
{
    Q_OBJECT

public:
    LFPExportHandler(QStringList arguments);
    int processData();
    ~LFPExportHandler();

protected:

    void parseArguments();
    void printHelpMenu();

    //void printCustomMenu();
    //void parseCustomArguments(int &argumentsProcessed);

private:
    bool useOneChannelPerNTrode;
    std::vector<int> channels;
    std::vector<int> tetrodes;
    std::vector<int> channelNumbers;
    int everything;

};

#endif // LFPEXPORTHANDLER_H

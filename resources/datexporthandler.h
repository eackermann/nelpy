#ifndef LFPEXPORTHANDLER_H
#define LFPEXPORTHANDLER_H
#include "abstractexporthandler.h"
#include "iirFilter.h"
#include "spikeDetectorThread.h"


class DATExportHandler: public AbstractExportHandler
{
    Q_OBJECT

public:
    DATExportHandler(QStringList arguments);
    int processData();
    ~DATExportHandler();

protected:

    void parseArguments();
    void printHelpMenu();

    //void printCustomMenu();
    //void parseCustomArguments(int &argumentsProcessed);

private slots:


private:
    bool invertSpikes;
    QStringList channelId;

};

#endif // LFPEXPORTHANDLER_H

# -*- coding: utf-8 -*-
"""
Created on Tue May  7 11:28:12 2019

@author: sanatara
"""
import pandas as pd
import os.path
import re
import glob, os
from enum import Enum, unique
import shutil

@unique
class PlotData(Enum):
    AcquisitionTime = 1
    ExecutionTime   = 2
    AcquisitionAndExecution = 3 

@unique    
class PlotType(Enum):
    LinePlot = 1
    BoxPlot = 2
    LineAndBoxPlots = 3

def ListFiles(fileRootPath, fileExtension):
    os.chdir(fileRootPath)
    fileList = []
    for file in glob.glob("*.txt"):
        fileList.append(file)
        print(file)
    return fileList

def CancatenateFiles(fileRoot):
    tempFilePath = fileRoot + "temp.txt"
    if os.path.exists(tempFilePath):
        os.remove(tempFilePath)
    fileList = ListFiles(fileRoot, "*.txt")
    with open(tempFilePath,'wb') as wfd:
        for f in fileList:
            with open(f,'rb') as fd:
                shutil.copyfileobj(fd, wfd)
    return tempFilePath 

######################################################################################################
# Configuration to plot Metrology Server Data
######################################################################################################                  
#  Root Path Metrology Log file to be plotted
#  INFO: All the file(s) in this folder will be merged into one 
#  WARNING: If tested outside IDE(Spyder), log files root path should have \\ instead of \
logFilesRoot ="D:\\next.SCADA\\Config\\Logs\\ManzMetrologyPickMeasurement\\Debug\\"
fileExtension = "\\*.txt"

# Select the data to be plotted: PlotData.
plotData = PlotData.AcquisitionTime

# Select the Plot types: PlotType.
plotType = PlotType.LineAndBoxPlots

#Title of the plot
plotTitle = "Acquisition Time_GV5280_With_SoftwareTrigger_Exposure_1ms"

# Path to save the plot
# Folder name will be automatically created.
# PlotTitle will be used as the file name
savePath ="D:/Tmp/Test"

# Set Flag to activate debug messages
detailedLogging = False
########################################################################################################

fileName = CancatenateFiles(logFilesRoot)
if (detailedLogging):
    print(fileName)

# Path of the file to be plotted
#fileName ="D:\\next.SCADA\\Config\\Logs\\ManzMetrologyPickMeasurement\\Debug\ManzDebug_ManzMetrologyPickMeasurement@DERTLP0696@_0.txt"

if plotData == PlotData.AcquisitionTime:
    searchKey = ': Acquisition'
elif plotData == PlotData.ExecutionTime:
    searchKey = ': Execution'
elif plotData == PlotData.AcquisitionAndExecution:
    searchKey = ''
else:
    print("{} not supported".format(plotData))
    
searchTag = searchKey.split(" ")[1] 

def CheckFolderExists(folderPath):
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)

def GeneratePlotFileName(folderPath, fileName):
    CheckFolderExists(folderPath)
    return os.path.join(os.path.normpath(folderPath), fileName)

def SavePlot(fig, csFileName):
    fig.savefig(csFileName, format='png', dpi=1200, bbox_inches = "tight")
    
def LinePlot():
     lines = dataFrame.plot.line(figsize=[7, 5], rot =90)
     lines.set_xlabel("Number of {} #".format(searchTag))
     lines.set_ylabel("{} Time [ms]".format(searchTag))
     Title = plotTitle
     lines.set_title(Title) 
     fig = lines.get_figure()
     fileName ="LinePlot_" + plotTitle + ".png"
     csFileName = GeneratePlotFileName(savePath, fileName)
     SavePlot(fig, csFileName)
     
def BoxPlot():
     boxPlot = dataFrame.boxplot(rot=90)
     Title =plotTitle
     boxPlot.set_title(Title)
     boxPlot.set_ylabel("{} Time [ms]".format(searchTag))
     fileName ="BoxPlot_" + plotTitle + ".png"
     csFileName = GeneratePlotFileName(savePath, fileName)
     fig = boxPlot.get_figure()
     SavePlot(fig, csFileName)

cycleTime = [] 
with open(fileName, 'r') as searchfile:
    for line in searchfile:
        if searchKey in line:
            time = re.search("{} (.+?) ms".format(searchTag), line)
            cycleTime.append(time.group(1))
            if detailedLogging:
                print('{} found {} ms'.format(searchTag, time.group(1)))
    print("INFO: {} number of entries found for the search key {}".format(len(cycleTime), searchTag))

if(len(cycleTime) > 0):
    columnName = '{} Time'.format(searchTag)
    dataFrame = pd.DataFrame(cycleTime, columns = [columnName] )
    columnValues = list(dataFrame.columns.values) 
    print(columnValues)
    convert_dict = {columnValues[0]: float } 
    dataFrame = dataFrame.astype(convert_dict) 
 
    if detailedLogging:
        print("data frame is:\n{}".format(dataFrame))
           
    if(plotType == PlotType.LinePlot):
         LinePlot()
    elif(plotType == PlotType.BoxPlot):
         BoxPlot()
    elif(plotType == PlotType.LineAndBoxPlots):   
        BoxPlot()
        LinePlot() 
    else:
        print("{} not supported".format(plotType))
        
    if os.path.exists(fileName):
        os.remove(fileName)
        
else:
    print("WARNING: No data for search key {} found in {}".format(searchKey,fileName))
#!/usr/bin/env python3

from PyQt5 import QtWidgets
from GUI_EGH455 import Ui_MainWindow
import sys, os, shutil
from PyQt5 import QtCore, QtMultimedia, QtMultimediaWidgets, QtWidgets, QtGui
import glob
import csv, codecs, cv2
import numpy as np
import datetime
from GUI_HelpWindow_Application import ApplicationWindow_Help
sys.path.insert(0, './retinanet')
import performDetection as PD
sys.path.insert(0, './yolonet')
import yolo_performDetection as YPD

# The class that handles the application itself
class ApplicationWindow(QtWidgets.QMainWindow):
    
    def __init__(self):
        # Handle the application display
        super(ApplicationWindow, self).__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.label_Ouput_Status.setText("Ready") # Update the status 
        self.dialogs = list() # For the second winow
        
        # Connect events (like button presses) to functions
        self.ui.button_play.clicked.connect(self.callback_play) # Play button         
        self.ui.button_play.setEnabled(False)
        self.ui.button_pause.clicked.connect(self.callback_pause) # Pause button 
        self.ui.button_pause.setEnabled(False)
        self.ui.button_stop.clicked.connect(self.callback_stop) # Stop button
        self.ui.button_stop.setEnabled(False)
        # self.ui.Help_toolButton.clicked.connect(self.callback_help) # Help button 
        self.ui.loadDataButton.clicked.connect(self.callback_load) # Load Data button
        self.ui.importCSVButton.clicked.connect(self.callback_impCSV) # Import CSV button
        self.ui.processDataButton.clicked.connect(self.callback_procData) # Process Data button
        self.ui.saveButton.clicked.connect(self.callback_save) # Save video button 
        self.ui.exportCSVButton.clicked.connect(self.callback_expCSV) # Export CSV button 
        self.ui.videoListWidget.itemClicked.connect(self.callback_playItem) # Video selected                 
        self.ui.videoListWidget.setEnabled(False) # Disable Video Chocie
        self.ui.DLMcomboBox.currentIndexChanged.connect(self.callback_DLMBox) # DLM selection 
        self.ui.csvListWidget.itemClicked.connect(self.callback_loadCSV) # Selected CSV file to load
        self.ui.csvListWidget.setEnabled(False) # Disable CSV Choice 
        self.ui.videoOnlyDataButton.clicked.connect(self.callback_videoOnlyDataButton) # Video analysing 
        self.ui.CSVOnlyDataButton.clicked.connect(self.callback_CSVOnlyDataButton) # CSV analysing 

        # Shortcuts 
        self.shortcut_play = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+p"), self) # Play Video
        self.shortcut_play.activated.connect(self.callback_play)
        self.shortcut_play.setEnabled(False)
        self.shortcut_pause = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+o"), self) # Pause Video
        self.shortcut_pause.activated.connect(self.callback_pause)
        self.shortcut_pause.setEnabled(False)
        self.shortcut_stop = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+k"), self) # Stop Video
        self.shortcut_stop.activated.connect(self.callback_stop)
        self.shortcut_stop.setEnabled(False)
        self.shortcut_Help = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+h"), self) # Help View
        self.shortcut_Help.activated.connect(self.callback_help)
        self.shortcut_load = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+l"), self) # Load Video
        self.shortcut_load.activated.connect(self.callback_load)
        self.shortcut_load.setEnabled(False)
        self.shortcut_impCSV = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+i"), self) # Import CSV 
        self.shortcut_impCSV.activated.connect(self.callback_impCSV)
        self.shortcut_impCSV.setEnabled(False)
        self.shortcut_saveVid = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+s"), self) # Save Video
        self.shortcut_saveVid.activated.connect(self.callback_save)
        self.shortcut_saveVid.setEnabled(False)
        self.shortcut_expCSV = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+e"), self) # Export CSV
        self.shortcut_expCSV.activated.connect(self.callback_expCSV)
        self.shortcut_expCSV.setEnabled(False)
        self.shortcut_EXIT = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+t"), self) # Exit
        self.shortcut_EXIT.activated.connect(self.close)

        # setup menu bar
        dirAction = QtWidgets.QAction("Select input directory", self)                
        dirAction.triggered.connect(self.callback_selectDir)

        refreshAction = QtWidgets.QAction("Refresh files", self)                
        refreshAction.triggered.connect(self.callback_refreshDir)

        helpAction = QtWidgets.QAction("Help", self)                
        helpAction.triggered.connect(self.callback_help)

        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')
        helpMenu = mainMenu.addMenu('Help')
        fileMenu.addAction(dirAction)
        fileMenu.addAction(refreshAction)
        helpMenu.addAction(helpAction)
        
        # Configure the video widgets
        self.video_player_input = QtMultimedia.QMediaPlayer(None, QtMultimedia.QMediaPlayer.VideoSurface)
        self.video_player_output = QtMultimedia.QMediaPlayer(None, QtMultimedia.QMediaPlayer.VideoSurface)
        # Configure the video selection 
        self.video_choice = None 
        # Configure the DLM selection 
        self.DLM = None
        self.ui.DLMcomboBox.addItem("RetinaNet")
        self.ui.DLMcomboBox.addItem("YOLO")
        # Configure the CSV selection 
        self.csv_choice = None 
        # Configure the CSV Data 
        self.csvFileArray = []
        # Configure Frame Number 
        self.frame_number = 0
        # Configure User Option if they want to full analyse video or CSV 
        self.Videoanalyse_CSVanalyse = None 
        # Configure number of objects
        self.number_sharks = 0
        self.number_dolhpins = 0 
        self.number_surfers = 0
        # Grab the file path
        self.file = None
        
        # Slider events 
        self.ui.horizontalSlider.sliderMoved.connect(self.callback_setPosition) # Slider manual moved 
        self.ui.horizontalSlider.setEnabled(False)

        # probe = QtMultimedia.QVideoProbe(self)
        # probe.videoFrameProbed.connect(self.positionChanged)
        # probe.setSource(self.video_player_input)
        self.video_player_input.positionChanged.connect(self.positionChanged) # Update slider based on video position
        self.video_player_input.durationChanged.connect(self.durationChanged) # Slider length based on video duration 

        # Search for all videos and CSV's in the current directory 
        self.inputDir = os.path.dirname(os.path.abspath(__file__))
        video_array = glob.glob('*.mp4')
        csv_array = glob.glob('*csv')

        # List View, display the videos to be selected 
        for i in video_array:
            self.ui.videoListWidget.addItem(i)
        
        # List View, display the CSV's to be selected 
        for j in csv_array:
            self.ui.csvListWidget.addItem(j)
            
        # Configure CSV
        self.delimit_frame = '\t'
        self.number_rows = 0     

    def callback_play(self):
        # Start video playback
        if self.video_player_input.state() == QtMultimedia.QMediaPlayer.PausedState:
            self.video_player_output.setPosition(self.video_player_input.position())        
            
        self.video_player_input.play()
        self.video_player_output.play()        
        self.ui.label_Ouput_Status.setText("Video Playing") # Update Status

    def callback_pause(self):
        # Pause video playback
        if self.video_player_input.state() == QtMultimedia.QMediaPlayer.PlayingState:
            self.ui.label_Ouput_Status.setText("Video Paused") # Update Status 
            self.video_player_input.pause()
            self.video_player_output.pause()
            self.video_player_output.setPosition(self.video_player_input.position())

    def callback_stop(self):
        # Pause video playback
        self.ui.label_Ouput_Status.setText("Video Stopped") # Update Status
        self.video_player_input.stop()
        self.video_player_output.stop()
    
    def callback_help(self):
        # Open the Help Button Class 
        self.help_child_win = ApplicationWindow_Help()
        self.help_child_win.show()
        
    def callback_load(self):    
        if self.ui.videoListWidget.selectedItems() != []:
            # Load in a file to play
            self.ui.label_Ouput_Status.setText("Loading Video") # Update Status
            self.file = os.path.join(self.inputDir, self.ui.videoListWidget.selectedItems()[0].text())        
            self.video_choice = self.file # Set the video choice made by the user 

            videoReader = cv2.VideoCapture(self.file)        
            self.fps = int(videoReader.get(cv2.CAP_PROP_FPS))    
            self.noFrames = videoReader.get(cv2.CAP_PROP_FRAME_COUNT)
            videoReader.release()
            self.video_player_input.setNotifyInterval(round((1000/self.fps)))

            self.video_player_input.setMedia(QtMultimedia.QMediaContent(QtCore.QUrl.fromLocalFile(self.file)))
            self.video_player_input.setVideoOutput(self.ui.video_widget_input)
            self.ui.video_widget_input.setAspectRatioMode(QtCore.Qt.KeepAspectRatioByExpanding)
            self.ui.label_Ouput_Status.setText("Video Loaded") # Update Status
        
            # Clear the CSV Statistics if it has previously been filled
            #self.ui.statsTableView.clearSpans()

            # enable play buttons and shortcuts
            self.ui.button_play.setEnabled(True)
            self.ui.button_pause.setEnabled(True)
            self.ui.button_stop.setEnabled(True)
            self.shortcut_play.setEnabled(True)
            self.shortcut_pause.setEnabled(True)
            self.shortcut_stop.setEnabled(True)
            self.ui.horizontalSlider.setEnabled(True)        
            
            # Enabled/Disable Buttons 
            # No CSV
            if self.Videoanalyse_CSVanalyse == 0:
                self.ui.processDataButton.setEnabled(True)
                self.ui.DLMcomboBox.setEnabled(True)
            else: 
                # CSV File
                self.ui.DLMcomboBox.setEnabled(False)
                self.ui.csvListWidget.setEnabled(True)             
    
    def callback_impCSV(self):
        # Load in a file to play
        self.csv_choice = os.path.join(self.inputDir, self.ui.csvListWidget.selectedItems()[0].text())        
        self.ui.label_Ouput_Status.setText("CSV Loaded") # Update Status

        # lock video controls
        # self.ui.button_play.setEnabled(False)
        # self.ui.button_pause.setEnabled(False)
        # self.ui.button_stop.setEnabled(False)
        
        # Open the CSV 
        # f = open(self.csv_choice, 'r')
        # mystring = f.read()
        # # Find each new frame 
        # if mystring.count(",") > mystring.count('\t'):
        #     if mystring.count(",") > mystring.count(';'):
        #         self.delimit_frame = ","
        #     elif mystring.count(";") > mystring.count(','):
        #         self.delimit_frame = ";"
        #     else:
        #         self.delimit_frame = "\t"
        # elif mystring.count(";") > mystring.count('\t'):
        #     self.delimit_frame = ';'
        # else:
        #     self.delimit_frame = "\t"

        # f.close()
        # Enable/Disable Buttons
        self.ui.processDataButton.setEnabled(True)
        
        
    def callback_procData(self):
        # Send the DLM and the video choice to the DEEP LEARNING Code 
        # Then grab the returned video and set up the output player 
        self.ui.label_Ouput_Status.setText("Processing Video") # Update Status
        
        # No CSV 
        if self.Videoanalyse_CSVanalyse == 0:
            print("Video Choice:" + str(self.video_choice))
            print("DLM Choice:" + str(self.DLM))
        else:
            # CSV
            print("Video Choice:" + str(self.video_choice))
            print("CSV Choice:" + str(self.csv_choice))

        progress = QtWidgets.QProgressDialog("Processing video ...", "", 0, self.noFrames, self)            
        progress.setWindowModality(QtCore.Qt.WindowModal)            
        progress.forceShow()
        progress.setCancelButton(None)                            
        progress.setValue(0)
        
        if self.Videoanalyse_CSVanalyse == 0:
            if self.DLM == 0:                
                out, csvOut = PD.retinanetDetection(self.file, progress)
            else:
                # yolo detection
                out, csvOut = YPD.yolonetDetection(self.file, progress)
            # Reset the CSV File Array
            self.csvFileArray = []
            self.csv_choice = csvOut
            self.output_csv_path = csvOut
        else:
            if self.DLM == 0:
                out = PD.overlayCSV(self.csv_choice, self.file, progress)
            else:
                out = YPD.overlayCSV(self.csv_choice, self.file, progress)
            self.csvFileArray = []
            self.output_csv_path = self.csv_choice
        
        out = QtCore.QDir.current().filePath(out) 
        self.output_video_path = out        
        self.video_player_output.setMedia(QtMultimedia.QMediaContent(QtCore.QUrl.fromLocalFile(out)))
        self.video_player_output.setVideoOutput(self.ui.video_widget_output)   
        
        # unlock video controls
        self.ui.button_play.setEnabled(True)
        self.ui.button_pause.setEnabled(True)
        self.ui.button_stop.setEnabled(True)       
        self.ui.saveButton.setEnabled(True)    
        self.ui.exportCSVButton.setEnabled(True)                         

        # Populate the CSV Statistics 
        f = open(self.csv_choice, 'r', encoding='utf-8')
        reader = csv.reader(f, delimiter = self.delimit_frame)
        next(reader, None)
        for row in reader:
            rowCSV = [int(row[0].split(',')[0]), row[0].split(',')[1].split()]
            self.csvFileArray.append(rowCSV)        
        f.close()
        self.update_csv() # Fill the CSV 
        
        # Enable/Disable Buttons 
        self.ui.processDataButton.setEnabled(False)
        self.ui.DLMcomboBox.setEnabled(False)
        self.ui.loadDataButton.setEnabled(False)
        self.shortcut_load.setEnabled(False)

        # Enable shortcuts
        self.shortcut_expCSV.setEnabled(True)
        self.shortcut_saveVid.setEnabled(True)
    
    def callback_save(self):
        # Save Video  
        filePath = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Video', os.path.dirname(os.path.realpath(__file__)), 'MP4(*.mp4)')
        if filePath[0] is not '':
            shutil.copy(self.output_video_path, filePath[0])                
    
    def callback_expCSV(self):
        # Export CSV  
        filePath = QtWidgets.QFileDialog.getSaveFileName(self, 'Save CSV', os.path.dirname(os.path.realpath(__file__)), 'CSV(*.csv)')
        if filePath[0] is not '':
            shutil.copy(self.output_csv_path, filePath[0])
    
    def positionChanged(self, position):
    # def positionChanged(self):
        # The video position has changed 
        # Theres 24 frames per second (frame 0 -> 23 = 1 second)
        self.frame_number = round((self.video_player_input.position()/1000)*self.fps)
        print("Frame Number: " + str(self.frame_number)) # DEBUGGING 
        
        # Get the current video time
        duration = str(datetime.timedelta(seconds=round(self.video_player_input.duration()/1000)))
        current_pos = str(datetime.timedelta(seconds=round(self.video_player_input.position()/1000)))
        
        # Update the video position label 
        self.ui.label_video_time.setText(current_pos +"/"+ duration)
                
        # NEED TO UPDATE THE csv_choice once the DLM returns the CSV file ###########################################################################
        # Each Change in the video position update the CSV acordingly 
        if (self.csv_choice != None): # Don't update CSV if only the original video is playing 
            self.update_csv()
        
        # Update the slider position based on video position 
        self.ui.horizontalSlider.setValue(position)        

    def durationChanged(self):
        # Update the slider proportions  
        self.ui.horizontalSlider.setRange(0, self.video_player_input.duration())
    
    def callback_setPosition(self):
        # Video Slider, set position based user defined 
        position = self.ui.horizontalSlider.value()
        self.video_player_input.setPosition(position)
        self.video_player_output.setPosition(position)
        
    def callback_playItem(self):
        if not self.ui.saveButton.isEnabled():
            self.ui.loadDataButton.setEnabled(True)
            self.shortcut_load.setEnabled(True)        
        print(self.ui.videoListWidget.currentRow())
        
    def callback_DLMBox(self):
        # Grab user selected DLM 
        self.DLM = self.ui.DLMcomboBox.currentIndex()
        
    def callback_loadCSV(self):
        self.ui.importCSVButton.setEnabled(True)
        self.shortcut_impCSV.setEnabled(True)
        print(self.ui.csvListWidget.currentRow())
        
    # def recieve_csv(self): 
    #     # When the DLM sends back the final CSV update the self.csvFileArray
    #     # Open the CSV 
    #     f = open(self.csv_choice, 'r')
    #     mystring = f.read()
    #     # Find each new frame 
    #     if mystring.count(",") > mystring.count('\t'):
    #         if mystring.count(",") > mystring.count(';'):
    #             self.delimit_frame = ","
    #         elif mystring.count(";") > mystring.count(','):
    #             self.delimit_frame = ";"
    #         else:
    #             self.delimit_frame = "\t"
    #     elif mystring.count(";") > mystring.count('\t'):
    #         self.delimit_frame = ';'
    #     else:
    #         self.delimit_frame = "\t"

    #     f.close()
    #     f = open(self.csv_choice, 'r', encoding='utf-8')
        
    #     for row in csv.reader(f, delimiter = self.delimit_frame):
    #         self.csvFileArray.append(row)
    #     self.csvFileArray = np.array(self.csvFileArray)
    #     print(self.csvFileArray[0])
    #     pass
        
    # Take the Frame number and grab the statistics       
    def update_csv(self):        
        # Hold the amount of Objects 
        sharks = 0
        dolphins = 0
        surfers = 0

        # Set up the statistics table by using a model 
        model = QtGui.QStandardItemModel(0, 6, self) # Set rows, columns 
        model.setHorizontalHeaderLabels(["Object", "Confidence %", "XMin", "YMin", "XMax", "YMax" ]) # Set labels
        
        if self.frame_number in [int(item[0]) for item in self.csvFileArray]:
            index = [int(item[0]) for item in self.csvFileArray].index(self.frame_number)                        
            for y in range(int(len(self.csvFileArray[index][1])/6)):
                append_data = []
                for v in range(6):
                    v = self.csvFileArray[index][1][y*6+v]
                    append_data.append(QtGui.QStandardItem(v))
                    if (v == "shark"):
                        sharks += 1
                    if (v == "dolphin"):
                        dolphins += 1
                    if (v == "surfer"):
                        surfers += 1
                model.appendRow(append_data)            

        # Update the object numbers 
        self.number_sharks = sharks 
        self.number_dolhpins = dolphins
        self.number_surfers = surfers
        
        # Display table and set labels 
        self.ui.statsTableView.setModel(model) 
        self.ui.statsTableView.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection) # Turn off row/columns selection
        self.ui.statsTableView.show()
        self.ui.statsTableView.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
        self.ui.label_Output_Surfers.setText(str(self.number_surfers))
        self.ui.label_Output_Dolphins.setText(str(self.number_dolhpins))
        self.ui.label_Output_Sharks.setText(str(self.number_sharks))                          

    def callback_videoOnlyDataButton(self):
        # revert to fresh
        revertFresh(self)

        # Enabled/Disable Buttons 
        self.Videoanalyse_CSVanalyse = 0
        self.ui.videoListWidget.setEnabled(True)                
    
    def callback_CSVOnlyDataButton(self):
        # revert to fresh
        revertFresh(self)
        
        # Enabled/Disable Buttons 
        self.Videoanalyse_CSVanalyse = 1
        self.ui.videoListWidget.setEnabled(True)                     

    def callback_selectDir(self):
        inputDir = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory Containing Video/CSV"))
        if inputDir is '':
            return
        self.inputDir = inputDir

        revertFresh(self)

        # Search for all videos and CSV's in the current directory         
        video_array = glob.glob(os.path.join(self.inputDir, '*.mp4'))
        csv_array = glob.glob(os.path.join(self.inputDir, '*.csv'))

        # List View, display the videos to be selected 
        self.ui.videoListWidget.clear()
        for i in video_array:
            self.ui.videoListWidget.addItem(os.path.basename(i))
        
        self.ui.csvListWidget.clear()
        # List View, display the CSV's to be selected 
        for j in csv_array:
            self.ui.csvListWidget.addItem(os.path.basename(j))    

    def callback_refreshDir(self):
        # Search for all videos and CSV's in the current directory         
        video_array = glob.glob(os.path.join(self.inputDir, '*.mp4'))
        csv_array = glob.glob(os.path.join(self.inputDir, '*.csv'))

        # List View, display the videos to be selected 
        self.ui.videoListWidget.clear()
        for i in video_array:
            self.ui.videoListWidget.addItem(os.path.basename(i))
        
        self.ui.csvListWidget.clear()
        # List View, display the CSV's to be selected 
        for j in csv_array:
            self.ui.csvListWidget.addItem(os.path.basename(j))  
    
def revertFresh(self):
        self.ui.videoListWidget.setEnabled(False)
        self.ui.videoListWidget.clearSelection()
        self.ui.csvListWidget.setEnabled(False)
        self.ui.csvListWidget.clearSelection()

        self.ui.button_play.setEnabled(False)
        self.ui.button_pause.setEnabled(False)
        self.ui.button_stop.setEnabled(False)
        self.ui.processDataButton.setEnabled(False)
        self.ui.saveButton.setEnabled(False)
        self.ui.exportCSVButton.setEnabled(False)
        self.ui.loadDataButton.setEnabled(False)
        self.ui.importCSVButton.setEnabled(False)

        self.ui.DLMcomboBox.setEnabled(False)

        self.ui.horizontalSlider.setEnabled(False)

        self.shortcut_play.setEnabled(False)
        self.shortcut_pause.setEnabled(False)
        self.shortcut_stop.setEnabled(False)        
        self.shortcut_load.setEnabled(False)
        self.shortcut_impCSV.setEnabled(False)        
        self.shortcut_saveVid.setEnabled(False)
        self.shortcut_expCSV.setEnabled(False)

        self.video_player_input.setMedia(QtMultimedia.QMediaContent())
        self.video_player_output.setMedia(QtMultimedia.QMediaContent())

        self.ui.statsTableView.setModel(None)
        self.csv_choice = None
        self.csvFileArray = None

        self.number_sharks = 0
        self.number_dolhpins = 0
        self.number_surfers = 0

        self.ui.label_Output_Surfers.setText(str(self.number_surfers))
        self.ui.label_Output_Dolphins.setText(str(self.number_dolhpins))
        self.ui.label_Output_Sharks.setText(str(self.number_sharks))  

# The "main()" function, like a C program
def main():
    print("Loading applicaiton...")
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    print("Application loaded.")
    application.show()
    sys.exit(app.exec_())

# Provides a start point for out code
if __name__ == "__main__":
    main()

## Installing PyQt5

PyQt5 is the default pyqt package for Python 3.6, and can be installed using Anoconda:
```sh
conda install -c anaconda pyqt
```

## Installing the rest of the QT Creator Suite
Follow this link to download the [QT Creator Suite](https://www1.qt.io/offline-installers/?hsLang=en#section-11) for your OS. The alternative is to download the entire [Qt Framework Development Kit](https://www.qt.io/download-qt-installer), in which _QT Designer_ is the used to allow graphical editting of the UI File.

## Preparing the Application
The proccess below utilises the files `gui_template.py` and `example_window.ui`, which are both found in this repository. Additionally, the file `gui_example.py` is provided as a reference of the completed python code that you should have by the end of this tutorial.

Before going further however, the `.ui` file must first be converted to a Python class (while it is not strictly necessary, it makes life a lot easier). To convert a `.ui` file to a Python class file, run the following command, replacing `FILENAME` with where appropriate (for the example, replace `FILENAME` with `example_window`):
```
python -m PyQt5.uic.pyuic -x FILENAME.ui -o FILENAME.py
```

#### Using the Python GUI Class
First, copy the code snippet below to be your program (or use the file `gui_template.py`), and name the new file something like `application.py`, making sure that the Python GUI class that was created earlier exists in the same folder:
```py
#!/usr/bin/env python3

from PyQt5 import QtWidgets
from FILENAME import Ui_MainWindow
import sys

# The class that handles the application itself
class ApplicationWindow(QtWidgets.QMainWindow):
	def __init__(self):
		# Create the Qt5 application backend
		super(ApplicationWindow, self).__init__()

		# Load in and display the UI
		self.ui = Ui_MainWindow()
		self.ui.setupUi(self)

# The "main()" function, like a C program
def main():
	print("Loading application...")
	app = QtWidgets.QApplication(sys.argv)
	application = ApplicationWindow()
	print("Application loaded.")
	application.show()
	sys.exit(app.exec_())

# Provides a start point for out code
if __name__ == "__main__":
	main()
```

Note that you will have to change `FILENAME` in the second line (`from FILENAME import Ui_MainWindow`) to match the filename used when you ran the `.ui` to `.py` conversion eariler.

At this point, you should be able to run your script and see your GUI displayed. When you exit the application, your script should also exit (and return to the command line):
```sh
python application.py
```

To import and use the GUI class (`FILENAME.py`), which defines the class `Ui_MainWindow`, first we have created a custom class `ApplicationWindow` that extends the class `PyQt5.QtWidgets.QMainWindow` (which provides the Qt5 application backend):
```py
# The class that handles the application itself
class ApplicationWindow(QtWidgets.QMainWindow):
	def __init__(self):
		# Create the Qt5 application backend
		super(ApplicationWindow, self).__init__()
```

Once the application is created, the UI (`Ui_MainWindow`) can then be loaded and shown:
```py
# Load in and display the UI
self.ui = Ui_MainWindow()
self.ui.setupUi(self)
```

The rest of the template handles the actual loading of the application in a standard python way (by creating `main()` which loads in the custom class `ApplicationWindow` and stops the script from exiting.

## Adding Application Functionallity
With the application started and displayed, the next step is to add some functionallity to the widgets on your application.

In the `ApplicationWindow` class, create 4 new callbacks, one for each button (you can reuse a callback for multiple button presses, but we will keep it simple for now). Each function must recieve `self` as an argument (allowing it to use class variables), but can be used to do anything you wish:
```py
class ApplicationWindow(QtWidgets.QMainWindow):
	def __init__(self):
		...

	def callback_button_1(self):
		print("Button 1 pressed!")

	def callback_button_2(self):
		print("Button 2 pressed!")

	def callback_button_3(self):
		print("Button 3 pressed!")

	def callback_button_4(self):
		print("Button 4 pressed!")
```

Next, in `__init__(self)` of `ApplicationWindow`, connecting the button 'clicked' event to the appropriate functions will allow for that function to be run each time the button is pressed:
```py
class ApplicationWindow(QtWidgets.QMainWindow):
	def __init__(self):
		# Create the Qt5 application backend
		...
		self.ui.setupUi(self)

		# Connect events (like button presses) to functions
		self.ui.button_1.clicked.connect(self.callback_button_1)
		self.ui.button_2.clicked.connect(self.callback_button_2)
		self.ui.button_3.clicked.connect(self.callback_button_3)
		self.ui.button_4.clicked.connect(self.callback_button_4)
```

Running the application again, you should now see messages printing to the terminal when you press a button.

## Designing a custom GUI

As previously mentioned, the file `example_window.ui` contains all of the inclusions and placment GUI elements for the application. To edit this easily, the program Qt Creator/Designer can be used. Assuming you installed on of them earlier on in the process, you should be able to simple double click `example_window.ui` to open QT Desginer and begin editing the file.

![QT Designer Interaface](https://raw.githubusercontent.com/qutas/egh455_gui_example/master/qt_designer.png)

The main features of the GUI designer include:
1. **Widget Box** - provides a selection of widgets that can be used in the GUI design. To add a new widget, simply drag it from the Widtet Box onto the form area. Some notable ones include:
   - [Label](http://doc.qt.io/qt-5/qlabel.html): Used to display text in the GUI
   - [Line Edit](http://doc.qt.io/qt-5/qlineedit.html): Allows a user to input a string that can be extracted later on
   - [Push Button](http://doc.qt.io/qt-5/qpushbutton.html): Acts as a regular button that can be used to call functions
   - [Combo Box](http://doc.qt.io/qt-5/qcombobox.html): Allows selection of a preset list of inputs
   - [Progress Bar](http://doc.qt.io/qt-5/qprogressbar.html): Can be used to display feedback to the user on function progress
   - Video Widget: (Not directly accessible through the QT Desginer GUI) Allows for a video to be interacted with through the GUI.
2. **Object Inspector** - provides access the objects that are present in your GUI. It is important to take note of the names of the objects that are added (and that you can change them to whatever you like), as you must use these names to access the GUI elements through python. For example, one of the buttons is named `button_1`, which is then access with the variable `self.ui.button_1` once the GUI is loaded into Python.
3. **Property Editor** - provides access to the properties of the currently selected object. Some useful examples are:
   - QObjectName: Sets the name of the
   - QWidget - Common widget properties:
     - `geometry`: used to set the GUI element location, as well as size
     - `sizePolicy`: Primarily used for setting the minimum size of the GUI form that will be allowed (i.e. makes sure that the GUI is rendered such that all the elements can be seen)
   - Other Properties - Depending on the widget, many other properties are available. For example, for a button, the QAbstractButton properties allow for things like `text` (the text displayed on the button) to be set.

####  Using Signals & Methods
As has been previously demonstrated, we can connect specific functionallity to cause updates to the GUI to trigger functions in our code to run. This is achieved by using the [Signal](http://doc.qt.io/qt-5/signalsandslots.html) interface in QT. Each widget has many types of signals that it emmits depending on how it is interacted with, so many so that a breif list cannot be provided. As an example however, let's look at some of the [primary signals available](http://doc.qt.io/qt-5/qabstractbutton.html#signals)with the QPushButton widget (which are inherited from the QAbstrctButton class):
- `clicked(bool checked = false)`: Triggered when a user presses and releases the button in a "normal" fashion.
- `pressed()`: Triggered whenever the button is depressed by the user.
- `released()`: Triggered whenever the button is released by the user.
- `toggled(bool checked)`: Triggered whenever the button state (pressed or released) changes.

As an example, the `clicked` signal is used in Python to attached a function to a button, such that it is run every time the `clicked` signal is triggered:
```py
self.ui.button_1.clicked.connect(self.callback_button_1)
```

#### Adding a Video Player (or Other Unlisted) Widget
Some widgets are not displayed by default in the QT Designer Widget Box. To access unlisted widgets, you must first add a blank widget to your form, then "promote" it to the type you want:
1. In the Wiget Box, navigate to the Containers section
2. Drag in a blank "Widget" widget to a location on your form
3. Right click on the blank widget, and select "Promote to ...", then set the following settings:
   - `Base class name`: `QWidget`
   - `Promote class name`: `QVideoWidget`
   - `Header file`: `PyQt5.QtMultimediaWidgets`
4. Press "Add", then press "Promote"
5. Select the blank widget, then in the properties editor, rename it to: `video_widget`
6. Save the `.ui` file, then regenerate the python file (See: Preparing the Application)
6. In your python application file, add the following import:
```py
from PyQt5 import QtCore, QtMultimedia
```
7. In your python application file, in the `__init__()`, add in the following code:
```py
# Configure the video widget
self.video_player = QtMultimedia.QMediaPlayer(None, QtMultimedia.QMediaPlayer.VideoSurface)

# Load in a file to play
file = QtCore.QDir.current().filePath("video.mp4")
self.video_player.setMedia(QtMultimedia.QMediaContent(QtCore.QUrl.fromLocalFile(file)))
self.video_player.setVideoOutput(self.ui.video_widget)

# Start video playback
self.video_player.play()
```

As a reference, the files `example_window_video.ui` and `gui_example_video.py` have been provided with a simple video widget implementation.

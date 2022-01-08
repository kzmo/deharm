# Deharm

Deharm is a Python application that let's you distort the spectrum of a sound
file by linearly shifting the frequencies so that the harmonic sounds become
unharmonic.

## Requirements

This software has been only tested on Python 3.9 and Windows 10. The operation
under any other Python version or operating system is unknown.

## Installation

If you haven't installed FFMPEG yet get it from [ffmpeg.org](https://www.ffmpeg.org/)

Clone Deharm from GitHub using Git from command line:

``` console
git clone https://github.com/kzmo/deharm.git
```

-OR-

Copy the Deharm directory into your python path. Zip
[here](https://github.com/kzmo/deharm/zipball/master)


In command line fetch the requirements
``` console
pip install -r requirements.txt
```

Now you should be able to run Deharm simply by running deharm.py:
``` console
python deharm.py
```

## Building a Windows stand alone executable

Deharm can be also built into a Windows stand alone .exe file. Follow the
installation instructions above and install also PyInstaller:
``` console
pip install pyinstaller
```

Then run the **build_windows.bat** script in the command line
``` console
build_windows.bat
```

The resulting **deharm.exe** file should be in the **win_build** directory.

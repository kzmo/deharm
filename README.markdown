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

# License

Copyright (c) 2022 Janne Valtanen (janne.valtanen@infowader.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE.

# Deharm

Deharm is a Python GUI application that lets you distort the spectrum of a sound
file by linearly shifting the frequencies so that the harmonic sounds become
unharmonic. This happens because the frequencies that used to be multiples of
each other get shifted so that they aren't any more.

## Requirements

This software requires at least Python version 3.8 or higher and it
has been only tested on Python 3.9 with Windows 10 and Fedora. The operation
under any other Python version or operating system is unknown.

Running the stand-alone Windows executable also requires FFMPEG. If you haven't
installed it yet get it from [ffmpeg.org](https://www.ffmpeg.org/). If
you don't want to install FFMPEG for the whole system you can just copy the
FFMPEG executable files (`ffmpeg.exe`, `ffplay.exe` and `ffprobe.exe` from the
`bin` directory) to the same directory as `deharm.exe`.

Running the software in Linux also requires an FFMPEG installation. Generally it
can be installed with whatever package manager is in use but for some
distributions you may need to add non-free repositories. For Fedora you need
the RPMfusion repositories. See
[here](https://docs.fedoraproject.org/en-US/quick-docs/setup_rpmfusion/).

## Windows stand-alone executable

The Windows stand-alone executable can be found in the releases
[here](https://github.com/kzmo/deharm/releases/download/v0.2.2/deharm.exe).

Please note the FFMPEG requirement described above in the requirements!

## Installation

Clone Deharm from GitHub using Git from command line:

``` console
git clone https://github.com/kzmo/deharm.git
```

-OR-

Copy the Deharm directory into your python path. The zip file can be found
[here](https://github.com/kzmo/deharm/zipball/master)


In the command line fetch the requirements in the main repository directory
``` console
pip install -r requirements.txt
```

-OR-

If `pip` is not in path you can install with Python command
``` console
python -m pip install -r requirements.txt
```

Now you should be able to run Deharm simply by running `deharm.py` from the
`src` directory:
``` console
python deharm.py
```

## Building a Windows stand-alone executable

Deharm can be also built into a Windows stand-alone .exe file. Follow the
installation instructions above and install also PyInstaller:
``` console
pip install pyinstaller
```

-OR-

``` console
python -m pip install pyinstaller
```

Then run the `build_windows.bat` script in the command line in the `build`
directory
``` console
build_windows.bat
```

The resulting `deharm.exe` file should be in the `build\win_build`
directory.

# Tips

The "Low Cut-off" option is used to set the minimum frequency where the effect
is applied on. This can be used for example to save the bassline and bass
drums from being shifted.

Normally frequencies are shifted to lower frequencies but "Shift to High"
switch can be used to shift the frequencies to higher frequencies.

If the processed sound has a jitter or a low frequency buzz, try to switch off
the "short-time FFT" option or try different FFT block sizes.

FFT block sizes that are powers of 2 are much faster. For example: 512, 1024,
2048.

Increasing the FFT block size increases the frequency resolution so that
the frequency options will match better to the actual frequencies.

Turning off the "short-time FFT" option will give you a very good frequency
resolution.

Without the "short-time FFT" option some of the frequencies might play
backwards due to phase reversal. This can be an interesting effect. But if it
is unwanted then turn on the "short-time FFT".

The experimental decomposition options can be used to apply the effects only on
harmonic or percussive components. The decomposition will slow down the
processing significantly though.

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

"""
.. module:: deharm.py
   :platform: Windows
   :synopsis: Deharm Kivy App

.. moduleauthor:: Janne Valtanen (janne.valtanen@infowader.com)
"""

import pathlib
import os
import tempfile
import datetime
import ctypes
import io
import os
import subprocess
import shutil

from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.factory import Factory
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.core.audio import SoundLoader
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from kivy.core.window import Window
from kivy.config import Config
import kivy
if kivy.utils.platform == 'win':
    # The Windows taskbar needs to be notified that this is a separate
    # app from Python to set the taskbar icon
    import ctypes
    myappid = 'deharm_gui.py'
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    # Imported just for the build to work
    import win32timezone

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
from matplotlib.ticker import EngFormatter

# Imported just for the build to work
import sklearn
import sklearn.neighbors._partition_nodes

from spectral_tools import *

# Version number
VERSION = "0.2"

# Last used file path as a string or NoneType
last_used_path = None

# Application configuration as a kivy.config.Config object or a NoneType
app_config = None

# A list of AudioData objects.
# The last one is the currently visible.
# Previous ones are the ones that can be reached by using the "undo" button
audio_datas = []


def get_default_directory():
    """Return the default directory in a string

    .. note:: Not tested on unix!

    Returns:
        (str): Default directory:
            Windows: %AppData%/deharm
            Unix: '~' expanded
    """

    if kivy.utils.platform == 'win':
        # In case this is running in Windows use a folder pointed to by APPDATA
        folder = os.path.join(os.getenv('APPDATA'), "deharm")
        if not os.path.isdir(folder):
            # If folder doesn't exist yet then create it.
            os.mkdir(folder)
        return folder
    else:
        return str(os.path.expanduser('~'))


def get_config_file_name():
    """Returns the configuration file full path

    Returns:
        (str): Full path to the configuration file
    """
    # Configuration is stored in the default directory/folder
    folder = get_default_directory()
    cfile = str(os.path.join(get_default_directory(), "deharm.ini"))
    return cfile


# Read the configuration file
Config.read(get_config_file_name())


def resource_path(relative_path):
    """Get absolute path to a resource based on the relative path.

    This is to make both the stand-alone executable and the script to work
    with file resources.

    Returns:
        (str): The absolute path
    """
    try:
        # PyInstaller temporary folder is stored in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # We are just running the app as a script. Get it from the
        # current directory.
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def prettify_time(t, pos=None):
    """Create a pretty time string in the format HH:MM:SS.f based on seconds

    Args:
        t(float): Time in seconds
        pos(None): Required by matplotlib for tick formatting

    Returns:
        (str): The time string
    """
    s = str(datetime.timedelta(seconds=t))
    sa = s.split('.')
    if t < 0:
        # HACK: Sometimes Kivy returns negative time when it should return 0
        return "0:00:00"
    elif len(sa) > 1:
        sa[-1] = sa[-1][:1]
        return '.'.join(sa)
    else:
        return s


class AudioData:
    """A single multi-channel audio data segment

    Contains the audio in floating point values and the associated metadata,
    spectrogram and Kivy audio

    .. note:: delete_temp_files should be called before losing reference!
    """

    def __init__(self, audio_data, sampling_frequency, nof_channels, length_s,
                 length_t, soundfile, on_playback_stop):
        """AudioData initializer function.

        Args:
            audio_data(numpy.ndarray): Audio data in a NumPy array
            sampling_frequency(float): Sampling frequency in Hz
            nof_channels(int): Number of channels
            length_s(float): Length in samples
            length_t(float): Length in seconds
            soundfile(str): The originating file's name
            on_playback_stop(function): Playback stop callback function
        """
        self.audio_data = audio_data
        self.sampling_frequency = sampling_frequency
        self.nof_channels = nof_channels
        self.length_s = length_s
        self.length_t = length_t
        self.temp_spectrogram = None
        self.soundfile = soundfile
        self.on_playback_stop = on_playback_stop
        self.temp_spectrogram = None

        # Create the audio for Kivy user interface
        self.create_kivy_audio()

    def create_kivy_audio(self):
        """Create the Kivy audio for the user interface
        """

        # Store the audio to a .wav file and reload to Kivy
        self.temp_audiofile = tempfile.NamedTemporaryFile(suffix=".wav",
                                                          delete=False)
        self.temp_audiofile.close()
        save_audio_data_to_wav(self.temp_audiofile.name,
                               self.audio_data,
                               int(self.sampling_frequency))
        self.kivy_sound = SoundLoader.load(self.temp_audiofile.name)

        # Set the callback for Kivy audio
        self.kivy_sound.on_stop = self.playback_stop_callback

    def playback_stop_callback(self):
        """The callback function when UI playback has stopped for this audio
        """

        # If playback was stopped then rewind

        # HACK: Kivy doesn't always rewind to exact zero so do it twice..
        self.kivy_sound.seek(0.0)
        self.kivy_sound.seek(0.0)

        # Call the playback stop callback defined in init
        self.on_playback_stop()

    def set_soundfile(self, soundfile):
        """Set the sound file filename

        Args:
            soundfile(str): Sound file's filename
        """
        self.soundfile = soundfile

    def save(self):
        """Save the audio to the file described by the self.soundfile
        """
        save_audio_data(self.soundfile, self.audio_data,
                        self.sampling_frequency)

    def play(self):
        """Play the audio via Kivy
        """
        self.kivy_sound.play()

    def stop(self):
        """Stop the Kivy playback
        """

        self.kivy_sound.stop()

    def get_playback_pos(self):
        """Get playback position in seconds

        Returns:
            (float): Playback position in seconds
        """
        return self.kivy_sound.get_pos()

    def get_length_t(self):
        """Get the audio length in seconds.

        Returns:
            (float): Audio length in seconds
        """
        return self.length_t

    def delete_temp_files(self):
        """Delete all temporary files created for this audio data
        """

        # Unload the Kivy audio from memory
        self.kivy_sound.unload()

        # Try to delete the temporary audio file for Kivy
        try:
            os.unlink(self.temp_audiofile.name)
        except FileNotFoundError:
            print("Temp file already deleted")

        # Try to delete the temporary spectrogram image
        try:
            if self.temp_spectrogram is not None:
                os.unlink(self.temp_spectrogram.name)
        except FileNotFoundError:
            print("Temp file already deleted")

    def generate_spectrogram(self, image_widget):
        """Generate the spectrogram and place it in a Kivy image widget

        Args:
            image_widget(kivy.uix.image.Image): The image widget where the
                spectrogram goes into
        """
        # Delete the old spectrogram if it still exists
        try:
            if self.temp_spectrogram is not None:
                os.unlink(self.temp_spectrogram.name)
                self.temp_spectrogram = None
        except FileNotFoundError:
            print("Temp file already deleted")

        # Get image size from the widget
        width_px = image_widget.size[0]
        height_px = image_widget.size[1]
        scale = 0.2
        dpi = height_px * scale
        plt.figure(figsize=(width_px/dpi, height_px/dpi), dpi=dpi)

        # Set colors for the spectrogram
        fg_color = "white"
        bg_color = "black"
        mpl.rcParams['text.color'] = fg_color
        mpl.rcParams['axes.labelcolor'] = fg_color
        mpl.rcParams['xtick.color'] = fg_color
        mpl.rcParams['ytick.color'] = fg_color
        mpl.rcParams['axes.edgecolor'] = fg_color
        mpl.rcParams["figure.facecolor"] = bg_color
        mpl.rcParams["figure.edgecolor"] = bg_color
        mpl.rcParams["savefig.facecolor"] = bg_color
        mpl.rcParams["savefig.edgecolor"] = bg_color

        # Plot spectrogram for each channel
        nof_channels = self.audio_data.shape[0]
        for channel, channel_number in zip(self.audio_data,
                                           range(nof_channels)):
            ha = plt.subplot(nof_channels, 1, channel_number + 1)
            plt.specgram(channel,
                         Fs=self.sampling_frequency,
                         scale='dB',
                         cmap="magma")
            ha.set_yscale('log')
            plt.ylim(20, self.sampling_frequency / 2)
            plt.xlim(0, self.length_t)
            plt.gca().yaxis.set_major_formatter(EngFormatter(unit='Hz'))
            plt.gca().xaxis.set_major_formatter(prettify_time)

        # Adjust the plot margins
        plt.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.06)

        # Save the spectrogram to a temporary .png file
        self.temp_spectrogram = tempfile.NamedTemporaryFile(suffix=".png",
                                                            delete=False)
        self.temp_spectrogram.close()
        plt.savefig(self.temp_spectrogram.name, format='png', dpi=dpi)

        # Load the temporary image file to the image widget
        image_widget.source = self.temp_spectrogram.name
        image_widget.reload()

    def reload_spectrogram(self, image_widget):
        """Reload the spectrogram image to a Kivy image widget

        Args:
            image_widget(kivy.uix.image.Image): The image widget where the
                spectrogram goes into
        """

        image_widget.source = self.temp_spectrogram.name
        image_widget.reload()

    def deharmonize(self, stft, fft_size, shift, minfreq, high, decompose):
        """Deharmonize the audio

        Args:
            stft(bool): Use short-time FFT?
            fft_size(str): FFT block size for short-time FFT
            shift(str): Linear shift in Hz
            minfreq(str): The low cut off of the shifted audio in Hz
            high(bool): Shift towards high frequencies?
            decompose(str): Decomposition type:
                "none" -- No decomposition
                "harmonic" -- Perform operation only on harmonic components
                "percussive" -- Perform operation only on percussive components
        """
        # Convert string values to numeric values
        fft_size = int(fft_size)
        shift = float(shift)
        minfreq = float(minfreq)

        if stft:
            # The short-time FFT case
            self.audio_data = deharmonize_stft(self.audio_data,
                                               self.sampling_frequency,
                                               shift,
                                               high,
                                               minfreq,
                                               decompose,
                                               fft_size)
        else:
            # The full-length FFT case
            self.audio_data = deharmonize(self.audio_data,
                                          self.sampling_frequency,
                                          shift,
                                          high,
                                          minfreq,
                                          decompose)

        # Unload and delete the old Kivy audio and update with new audio
        self.kivy_sound.unload()
        try:
            os.unlink(self.temp_audiofile.name)
        except FileNotFoundError:
            print("Temp file already deleted")
        self.create_kivy_audio()

    def __del__(self):
        """Delete temporary files when garbage collected
        """
        self.delete_temp_files()


class LoadDialog(FloatLayout):
    """The file load dialog window
    """
    # Load button
    load = ObjectProperty(None)
    # Cancel button
    cancel = ObjectProperty(None)

    def default_path(self):
        """Returns the default path to be used.

        Returns:
            (str): the last used path if defined. Otherwise the
                parent file's path.
        """
        if last_used_path is None:
            return str(pathlib.Path(__file__).parent.resolve())
        else:
            return last_used_path


class SaveDialogContent(FloatLayout):
    """Save file dialog window
    """
    # Save button
    save = ObjectProperty(None)
    # Cancel button
    cancel = ObjectProperty(None)


class SaveAsDialog(FloatLayout):
    """Save As dialog window
    """
    # Save button
    save = ObjectProperty(None)
    # File name field
    text_input = ObjectProperty(None)
    # Cancel button
    cancel = ObjectProperty(None)

    def default_path(self):
        """Returns the default path to be used.

        Returns:
            (str): The last used path if defined. Otherwise the
                parent file's path.
        """
        if last_used_path is None:
            return str(pathlib.Path(__file__).parent.resolve())
        else:
            return last_used_path


class SaveErrorDialog(FloatLayout):
    """Save error dialog window
    """
    # Cancel button
    cancel = ObjectProperty(None)
    # The error message
    error_msg = ObjectProperty(None)


class LoadErrorDialog(FloatLayout):
    """Load error dialog window
    """
    # Cancel button
    cancel = ObjectProperty(None)
    # The error message
    error_msg = ObjectProperty(None)


class ProcessErrorDialog(FloatLayout):
    """Process error dialog window
    """
    # Cancel button
    cancel = ObjectProperty(None)
    # Error message
    error_msg = ObjectProperty(None)


class SplashScreen(Popup):
    """Splash screen dialog window
    """
    # Close button
    cancel = ObjectProperty(None)
    # Software version
    version = ObjectProperty(None)
    # Splash screen text from LICENSE
    with open(resource_path("LICENSE"), "r") as license:
        text = license.read()


class Root(FloatLayout):
    """Root window layout as a Kivy FloatLayout
    """

    def __init__(self, **kwargs):
        """Standard initializer
        """

        super().__init__(**kwargs)

        # Playback's don't have updater callback by default
        self.playback_updater = None

        # Saving without approving is by default off
        self.save_approved = False

        # Scheduled resize is not defined by default
        self.resize_scheduler = None

        # Set the window resize callback
        Window.bind(on_resize=self.on_window_resize)

    def get_version(self):
        """Returns the software version

        Returns:
            (str): Version string
        """
        return VERSION

    def on_window_resize(self, window, width, height):
        """Window resize callback

        Triggers a scheduled event 2 seconds later to update the spectrogram.
        This is to prevent a flood of spectrogram resizes.

        Args:
            window(kivy.core.window.window_sdl2.WindowSDL): The resized window
            width(int): The new width in pixels
            height(int): The new height in pixels
        """
        # If there was an old scheduler already then cancel it
        if self.resize_scheduler is not None:
            self.resize_scheduler.cancel()
        # Schedule the spectrogram resize to happen in 2 seconds
        self.resize_scheduler = Clock.schedule_once(self.resize_spectrogram, 2)

    def resize_spectrogram(self, *largs):
        """Resize the visible spectrogram
        """
        # The last item in the audio_datas list is the visible spectrogram
        if len(audio_datas) > 0:
            audio_datas[-1].generate_spectrogram(self.ids.spectrogram)

    def get_nofile_img(self):
        """Return the "no file" image in case there is no audio yet loaded.
        """
        return resource_path("nofile.png")

    def update_title(self, filename=None):
        """Update the window title based on filename

        Args:
            filename(str or NoneType): The name of the file currently being
                edited.
        """
        app = App.get_running_app()
        app.update_title(filename)

    def hide_splash_screen(self):
        """Hide the splash screen dialog window
        """
        self.remove_widget(self.ids.splashscreen)

    def dismiss_popup(self):
        """Dismiss a popup window
        """
        self._popup.dismiss()

    def show_load(self):
        """Show the load file dialog window
        """
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def show_save_as(self):
        """Show the save as dialog window
        """
        content = SaveAsDialog(save=self.save, cancel=self.dismiss_popup)
        self._popup = Popup(title="Save file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def load(self, path, filename):
        """Load a new sound file

        Args:
            path(str): The path to directory where the file is located
            filename(list): List of filenames (Only the first one is used)
        """
        global last_used_path
        global audio_datas

        # Generate the full path to the sound file
        soundfile = os.path.join(path, filename[0])

        # Stop the audio playback if there is previous audio loaded
        if getattr(self, "playback_updater", None) is not None:
            self.stop_audio()

        def playback_stop_callback():
            """Playback stop callback

            Just disables the stop button
            """
            self.ids.stop_button.disabled = True

        try:
            # Empty the audio datas list and start with a new List
            # containing only the loaded audio
            new_audio_datas = [AudioData(*get_audio_data(soundfile),
                                         soundfile,
                                         lambda: playback_stop_callback())]

            # Delete the temp files from old audio datas
            for audio_data in audio_datas:
                audio_data.delete_temp_files()
            audio_datas = new_audio_datas

        except Exception as ex:
            # In case something went wrong show an error message with the
            # exception information
            self.dismiss_popup()
            csize = 80
            og_txt = str(ex)
            error_msg = [og_txt[i: i + csize] for i in range(0, len(og_txt),
                                                             csize)]
            error_msg = "\n" + "\n".join(error_msg)
            popup_content = LoadErrorDialog(cancel=self.dismiss_popup,
                                            error_msg=error_msg,
                                            size_hint=(0.9, 0.9))
            self._popup = Popup(title="Load Error", content=popup_content,
                                auto_dismiss=False)
            self._popup.open()
            return

        # Update the last used path
        last_used_path = path
        app_config.set("General", "default_path", last_used_path)

        # Generate spectrogram for UI
        audio_datas[-1].generate_spectrogram(self.ids.spectrogram)

        # Set position text for UI
        self.ids.playback_pos_text.text = \
            self.get_position_text(0, audio_datas[-1].get_length_t())
        self.ids.playback_progress.value = 0.0

        # Enable buttons in UI
        self.save_approved = False
        self.ids.save_button.disabled = False
        self.ids.save_as_button.disabled = False
        self.ids.play_button.disabled = False
        self.ids.deharmonize_button.disabled = False

        # Update the window title with the filename
        self.update_title(filename=os.path.split(filename[0])[-1])

        # Hide the open file dialog
        self.dismiss_popup()

    def save(self, path=None, filename=None):
        """Save the current audio to a file

        Args:
            path(str or NoneType): Path to the file.
                NoneType if saving to the last used file.
            filename(str or NoneType): Filename.
                NoneType if saving to the last used file.
        """
        global audio_datas
        global last_used_path

        # Dismiss any dialog window
        self.dismiss_popup()
        try:
            if filename is None:
                # No new filename save to the old one
                audio_datas[-1].save()
            else:
                # A new filename.
                soundfile = os.path.join(path, filename)
                audio_datas[-1].set_soundfile(soundfile)
                audio_datas[-1].save()
                # New files are always approved for saving.
                self.save_approved = True
                last_used_path = path
                app_config.set("General", "default_path", last_used_path)
                # Update the window title.
                self.update_title(filename=filename)
        except Exception as ex:
            # Something went wrong. Show the error dialog with the exception
            # information.
            popup_content = SaveErrorDialog(cancel=self.dismiss_popup,
                                            error_msg=str(ex),
                                            size_hint=(0.9, 0.9))
            self._popup = Popup(title="Save Error",
                                content=popup_content,
                                auto_dismiss=False)
            self._popup.open()

    def save_clicked(self):
        """Callback when Save button is clicked

        Opens a confirmation dialog window if direct saves are not yet
        approved
        """
        if not self.save_approved:
            popup_content = SaveDialogContent(save=self.save,
                                              cancel=self.dismiss_popup,
                                              size_hint=(0.9, 0.9))
            self._popup = Popup(title="Save",
                                content=popup_content,
                                auto_dismiss=False)
            self._popup.open()
            self.save_approved = True
        else:
            self.save()

    def play_audio(self):
        """Play the audio from the last item in audio_datas
        """
        global audio_datas
        global playback_updater

        # Play the last item in audio_datas
        audio_datas[-1].play()

        # Set the playback updates to happen every 0.1s to the UI
        self.playback_updater = \
            Clock.schedule_interval(self.playback_updater_callback, 0.1)

        # Enable the stop button
        self.ids.stop_button.disabled = False

    def stop_audio(self):
        """Stop audio playback
        """
        global audio_datas
        global playback_updater
        audio_datas[-1].stop()
        # Stop the playback update to the UI

        Clock.unschedule(self.playback_updater)
        # Schedule one extra update to UI to get correct time position

        Clock.schedule_once(self.playback_updater, 0.1)

        # Enable the stop button
        self.ids.stop_button.disabled = True

    def get_position_text(self, playback_pos, length_t):
        """Generate the playback position text for UI

        Args:
            playback_pos(float): Playback position in seconds
            length_t(float): Total audio length in seconds

        Returns:
            (str): String containing elapsed/total time
        """
        global audio_datas
        return f"{prettify_time(playback_pos)} / {prettify_time(length_t)}"

    def playback_updater_callback(self, *largs):
        """Playback update callback

        Updates the playback position text and the progress bar
        """
        playback_pos = audio_datas[-1].get_playback_pos()
        length_t = audio_datas[-1].get_length_t()
        self.ids.playback_pos_text.text = self.get_position_text(playback_pos,
                                                                 length_t)
        self.ids.playback_progress.value = 100.0 * playback_pos / length_t

    def undo(self):
        """Undo the last deharmonization

        Destroys the latest deharmonized audio data and updates the UI
        """
        global audio_datas

        # Stop audio if it was playing
        if getattr(self, "playback_updater", None) is not None:
            self.stop_audio()

        # Delete the temporary files from the latest audio
        audio_datas[-1].delete_temp_files()

        # Drop the latest audio data from the list
        audio_datas = audio_datas[:-1]

        # Update playback position in UI
        self.ids.playback_pos_text.text = \
            self.get_position_text(0, audio_datas[-1].get_length_t())
        self.ids.playback_progress.value = 0.0

        # Reload the spectrogram
        audio_datas[-1].reload_spectrogram(self.ids.spectrogram)

        # Disable the undo button if there is only one audio left
        if len(audio_datas) <= 1:
            self.ids.undo.disabled = True

    def decompose_none_clicked(self):
        """Callback when "None" option is clicked for decomposition

        Set the decompose values in UI accordingly
        """
        value = self.ids.decompose_none.active
        if value:
            self.ids.decompose_harmonic.active = False
            self.ids.decompose_percussive.active = False
        else:
            self.ids.decompose_none.active = True

    def decompose_harmonic_clicked(self):
        """Callback when "Harmonic" option is clicked for decomposition

        Set the decompose values in UI accordingly
        """
        value = self.ids.decompose_harmonic.active
        if value:
            self.ids.decompose_none.active = False
            self.ids.decompose_percussive.active = False
        else:
            self.ids.decompose_harmonic.active = True

    def decompose_percussive_clicked(self):
        """Callback when "Percussive" option is clicked for decomposition

        Set the decompose values in UI accordingly
        """
        value = self.ids.decompose_percussive.active
        if value:
            self.ids.decompose_harmonic.active = False
            self.ids.decompose_none.active = False
        else:
            self.ids.decompose_percussive.active = True

    def schedule_deharmonize(self):
        """Schedule a deharmonization

        Deharminization is scheduled in the future to update the spectrogram
        image to "Processing" text while the deharmonization happens
        """
        # Set the processing text while the transformation happens
        self.ids.spectrogram.source = resource_path("processing.png")
        self.ids.spectrogram.reload()

        # Schedule the actual deharmonization.
        Clock.schedule_once(self.deharmonize, 0.01)

    def deharmonize(self, *largs):
        """Deharmonize the latest audio based on UI options
        """
        global audio_datas

        # Create a copy of the last audio data
        if getattr(self, "playback_updater", None) is not None:
            self.stop_audio()
        last = audio_datas[-1]
        audio_datas.append(AudioData(last.audio_data, last.sampling_frequency,
                                     last.nof_channels, last.length_s,
                                     last.length_t, last.soundfile,
                                     last.on_playback_stop))

        # Get the decomposition type
        if self.ids.decompose_harmonic.active:
            decompose = "harmonic"
        elif self.ids.decompose_percussive.active:
            decompose = "percussive"
        else:
            decompose = "none"

        # Deharmonize the new copy
        try:
            audio_datas[-1].deharmonize(stft=self.ids.stft_checkbox.active,
                                        fft_size=self.ids.fft_size.text,
                                        shift=self.ids.shift.text,
                                        minfreq=self.ids.minfreq.text,
                                        high=self.ids.high_checkbox.active,
                                        decompose=decompose)
        except Exception as ex:
            # Something went wrong. Show the error dialog with the exception
            # information.
            csize = 80
            og_txt = str(ex)
            error_msg = [og_txt[i: i + csize] for i in range(0, len(og_txt),
                                                             csize)]
            error_msg = "\n" + "\n".join(error_msg)
            popup_content = ProcessErrorDialog(cancel=self.dismiss_popup,
                                               error_msg=error_msg,
                                               size_hint=(0.9, 0.9))
            self._popup = Popup(title="Process Error", content=popup_content,
                                auto_dismiss=False)
            self._popup.open()
            # Remove the last audio data
            audio_datas[-1].delete_temp_files()
            audio_datas = audio_datas[:-1]
            # Reload the spectrogram
            audio_datas[-1].reload_spectrogram(self.ids.spectrogram)
            return

        # Update the spectrogram
        audio_datas[-1].generate_spectrogram(self.ids.spectrogram)

        # Set playback position to 0
        self.ids.playback_pos_text.text = \
            self.get_position_text(0, audio_datas[-1].get_length_t())
        self.ids.playback_progress.value = 0.0

        # Enable the undo button
        self.ids.undo.disabled = False


class DeharmApp(App):
    """The Deharm Kivy App
    """
    def build_config(self, config):
        """Build the app configuration

        config(kivy.config.Config): The configuration for this app
        """
        # The general Config is used because it contains the correct filename
        Config.setdefaults('General', {
            "default_path": get_default_directory()
        })

    def build(self):
        """Kivy app build function.

        Called when the app is run.
        """
        global app_config
        global last_used_path

        # Set icon for the app window
        self.icon = resource_path('deharm_icon_256.png')

        # Create the app configuration
        app_config = Config

        # Get the last used path from configuration if available
        last_used_path = app_config.get("General", "default_path")

        # Update the window title
        self.update_title()

    def update_title(self, filename=None):
        """Update the window title

        Args:
            filename(str or NoneType): The filename included in the window
                title if available
        """
        title = f"Deharm v{VERSION}"
        if filename is None:
            self.title = title
        else:
            self.title = title + f" ({filename})"

    def on_stop(self):
        """App stop callback
        """
        global audio_datas

        # Write configuration
        Config.write()

        # Clean up the audio datas
        for audio_data in audio_datas:
            audio_data.delete_temp_files()

        audio_datas = []


if __name__ == "__main__":
    # Register all UI component classes
    Factory.register('Root', cls=Root)
    Factory.register('LoadDialog', cls=LoadDialog)
    Factory.register('SaveDialogContent', cls=SaveDialogContent)
    Factory.register('SaveAsDialog', cls=SaveAsDialog)
    Factory.register('SaveErrorDialog', cls=SaveErrorDialog)
    Factory.register('LoadErrorDialog', cls=LoadErrorDialog)
    Factory.register('ProcessErrorDialog', cls=ProcessErrorDialog)
    Factory.register('SplashScreen', cls=SplashScreen)

    # Check if FFMPEG is available and if not then just show error message
    # on screen.
    if shutil.which("ffmpeg") is None:
        class ErrorApp(App):
            def build(self):
                text = "FFMPEG not found!\n\n" \
                       "Please install FFMPEG before running this program!\n"\
                       "\n" \
                       "See https://www.ffmpeg.org/"
                content = TextInput(text=text)
                return Popup(title="Error!", content=content)
        ErrorApp().run()
    else:
        # FFMPEG found. Continue with the normal app.
        DeharmApp().run()

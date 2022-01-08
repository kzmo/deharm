python -m PyInstaller --onefile --noconfirm --noconsole deharm.spec
del win_build
mkdir win_build
copy dist\deharm.exe win_build\

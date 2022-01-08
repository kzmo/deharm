# -*- mode: python ; coding: utf-8 -*-
from kivy_deps import sdl2, glew
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None


a = Analysis(['..\\src\\deharm.py'],
             pathex=[],
             binaries=[],
             datas=collect_data_files('librosa'),
             hiddenimports=[],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

a.datas += [('deharm_icon_256.png', '..\\src\\deharm_icon_256.png', "DATA")]
a.datas += [('processing.png', '..\\src\\processing.png', "DATA")]
a.datas += [('nofile.png', '..\\src\\nofile.png', "DATA")]
a.datas += [('LICENSE', '..\\src\\LICENSE', "DATA")]

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)


exe = EXE(pyz, Tree('..\\src'),
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          *[Tree(p) for p in (sdl2.dep_bins + glew.dep_bins)],
          upx=True,
          console=False,
          icon="deharm.ico",
          name='deharm')

coll = COLLECT(exe, Tree('..\\src'),
               a.binaries,
               a.zipfiles,
               a.datas,
               *[Tree(p) for p in (sdl2.dep_bins + glew.dep_bins)],
               strip=False,
               upx=True,
               name='deharm')

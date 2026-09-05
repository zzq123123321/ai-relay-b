# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for AI Relay B (fixed build).

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

hiddenimports = []
hiddenimports += collect_submodules("comtypes")
hiddenimports += collect_submodules("uiautomation")

datas = []
datas += collect_data_files("uiautomation")
datas += collect_data_files("comtypes")

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AI_Relay_B",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="AI_Relay_B",
)

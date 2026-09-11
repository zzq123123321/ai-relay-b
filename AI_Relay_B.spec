# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for AI Relay B (fixed build).

from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = [
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "comtypes.client",
    "comtypes.stream",
]
for package in ("uiautomation", "comtypes"):
    package_data, package_binaries, package_imports = collect_all(package)
    datas += package_data
    binaries += package_binaries
    hiddenimports += package_imports

# 过滤从 Codex/Poppler 误收集进来的原生 DLL。
# PySide6 的 Qt6Core.dll 依赖 icuuc.dll，但打包工具在 DLL 搜索过程中
# 误取了 Codex 运行时（poppler/libheif）目录下的病态 DLL：
#   - poppler 的 icuuc.dll/icudt78.dll：ICU 版本/构建与 Qt6 不匹配，
#     导致 QtCore 加载时“找不到指定的程序”（DLL load failed）。
#   - poppler 的 libcrypto-3-x64.dll / libssl-3-x64.dll：与 Python 自带的
#     libcrypto-3.dll/libssl-3.dll 同名冲突。
# 这些 DLL 不属于本应用，必须排除。Qt6Core 会回退到系统 ICU（与旧版一致）。
CODEX_DLL_PREFIX = "codex-runtimes\\codex-primary-runtime\\dependencies\\native"
problematic_extensions = (
    "icuuc.dll",
    "icudt78.dll",
    "icuin.dll",
    "libcrypto-3-x64.dll",
    "libssl-3-x64.dll",
)
binaries = [
    (dest, src)
    for (dest, src) in binaries
    if not (
        CODEX_DLL_PREFIX in src
        and any(dest.lower() == ext.lower() for ext in problematic_extensions)
    )
]


def _is_problematic_binary(dest: str, src: str) -> bool:
    return CODEX_DLL_PREFIX in src and any(
        dest.lower() == ext.lower() for ext in problematic_extensions
    )


a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

# PyInstaller's dependency resolution (bindepend) re-adds the Codex/Poppler
# ICU/OpenSSL DLLs discovered on the search path even when they are absent
# from the input ``binaries`` list (they end up in a.binaries).  Filter the
# resolved list as well so the running app never ships them.
a.binaries = [
    entry
    for entry in a.binaries
    if not (
        len(entry) >= 2 and _is_problematic_binary(entry[0], entry[1])
    )
]

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

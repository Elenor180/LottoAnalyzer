# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['LottoApp2.py'],
    pathex=[],
    binaries=[],
    datas=[('lotto_history.csv', '.'), ('generated_combos.csv', '.')],
    hiddenimports=['sklearn.cluster._kmeans', 'mlxtend.frequent_patterns', 'mlxtend.preprocessing', 'tensorflow', 'tensorflow.keras', 'tensorflow.keras.models', 'tensorflow.keras.layers'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='LottoAnalyzerApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

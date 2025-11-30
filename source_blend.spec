# -*- mode: python ; coding: utf-8 -*-


block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'moderngl',
        'glcontext',
        'srctools.vtf',
        'srctools.vpk',
        'srctools.vmt',
        'PIL._imaging',
        'PIL.Image',
        'PIL.ImageColor',
        'typer.main',
        'typer.core',
        'typer.models',
        'click',
        'pygments'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unused stdlib modules
        'tkinter',
        'unittest',
        'test',
        'distutils',
        'email',
        'html',
        'http',
        'xml',
        'xmlrpc',
        'pydoc',
        'doctest',
        'argparse',
        'tarfile',
        'bz2',
        '_bz2',
        'sqlite3',
        '_sqlite3',
        # Exclude unused numpy components
        'numpy.distutils',
        'numpy.f2py',
        'numpy.testing',
        # Exclude rich/typer unnecessary deps
        'IPython',
        'jedi',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Remove unnecessary binaries
a.binaries = [x for x in a.binaries if not x[0].startswith('_decimal')]
a.binaries = [x for x in a.binaries if not x[0].startswith('_hashlib')]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='source_blend',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

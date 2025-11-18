# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

datas = [
    ('data', 'data'),
    ('models', 'models'),
    ('frame_actions', 'frame_actions'),
    ('styles', 'styles'),
    ('logo_muni.jpg', '.'),
    ('users.db', '.'),
]

# incluye también el archivo QSS directamente
datas += [('styles/main.qss', 'styles')]

datas += [
    ('mediapipe_data/modules/hand_landmark', 'mediapipe/modules/hand_landmark'),
    ('mediapipe_data/modules/palm_detection', 'mediapipe/modules/palm_detection'),
    ('mediapipe_data/modules/holistic_landmark', 'mediapipe/modules/holistic_landmark'),
    ('mediapipe_data/modules/pose_landmark', 'mediapipe/modules/pose_landmark'),
    ('mediapipe_data/modules/face_landmark', 'mediapipe/modules/face_landmark'),

]

hiddenimports = [
    'mediapipe.python.solutions.hands',
    'mediapipe.python.solutions.holistic',
    'mediapipe.framework.formats.landmark_pb2',
    'mediapipe.framework.formats.detection_pb2',
    'mediapipe.framework.formats.classification_pb2',
    'mediapipe.framework.formats.rect_pb2',
    'mediapipe.framework.formats.normalized_landmark_pb2',
    'sklearn.pipeline',
    'sklearn.preprocessing',
    'sklearn.linear_model',
    'sklearn.utils._weight_vector',
]

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='SingApp',
    debug=False,
    strip=False,
    upx=False,
    console=False,   
    icon='icon.ico', 
)

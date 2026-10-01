import os
import struct
import numpy as np
from collections import namedtuple

# Minimal mapping for COLMAP camera model IDs to number of parameters.
CAMERA_MODEL_PARAMS = {
    0: 3,   # SIMPLE_PINHOLE (f, cx, cy)
    1: 4,   # PINHOLE (fx, fy, cx, cy)
    2: 4,   # SIMPLE_RADIAL (f, cx, cy, k)
    3: 5,   # RADIAL (f, cx, cy, k1, k2)
    4: 8,   # OPENCV
    5: 8,   # OPENCV_FISHEYE
    6: 10,  # FULL_OPENCV
    7: 5,   # FOV
    8: 4,   # SIMPLE_RADIAL_FISHEYE
    9: 5,   # RADIAL_FISHEYE
    10: 12 # THIN_PRISM_FISHEYE
}

Camera = namedtuple('Camera', ['id', 'model', 'width', 'height', 'params'])
# Extend Image to carry point3D IDs (list of ints)
Image = namedtuple('Image', ['id', 'qvec', 'tvec', 'camera_id', 'name', 'point3d_ids'])
Point3D = namedtuple('Point3D', ['id', 'xyz', 'rgb', 'error'])

def _read_cameras_binary(path: str):
    cameras = {}
    with open(path, 'rb') as fid:
        num_cameras = struct.unpack('<Q', fid.read(8))[0]
        for _ in range(num_cameras):
            cam_id, model_id, width, height = struct.unpack('<iiQQ', fid.read(24))
            param_num = CAMERA_MODEL_PARAMS.get(model_id, 0)
            params = struct.unpack('<' + 'd' * param_num, fid.read(8 * param_num))
            cameras[cam_id] = Camera(id=cam_id, model=model_id, width=width, height=height, params=params)
    return cameras

def _read_images_binary(path: str):
    images = {}
    with open(path, 'rb') as fid:
        num_images = struct.unpack('<Q', fid.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack('<i', fid.read(4))[0]
            qvec = struct.unpack('<dddd', fid.read(32))
            tvec = struct.unpack('<ddd', fid.read(24))
            camera_id = struct.unpack('<i', fid.read(4))[0]
            # name is a null‑terminated string
            name_bytes = []
            while True:
                c = fid.read(1)
                if c == b'\x00' or c == b'':
                    break
                name_bytes.append(c)
            name = b''.join(name_bytes).decode('utf-8')
            # Number of 2D points observed in this image
            num_points2D = struct.unpack('<Q', fid.read(8))[0]
            point3d_ids = []
            for _ in range(num_points2D):
                # each point: x (double), y (double), point3D_id (int64)
                fid.read(8 * 2)  # skip x, y
                pid = struct.unpack('<q', fid.read(8))[0]
                point3d_ids.append(pid)
            images[image_id] = Image(id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id, name=name, point3d_ids=point3d_ids)
    return images

def _read_points3d_binary(path: str):
    points3D = {}
    with open(path, 'rb') as fid:
        num_points = struct.unpack('<Q', fid.read(8))[0]
        for _ in range(num_points):
            pt_id = struct.unpack('<q', fid.read(8))[0]
            xyz = struct.unpack('<ddd', fid.read(24))
            rgb = struct.unpack('<BBB', fid.read(3))
            error = struct.unpack('<d', fid.read(8))[0]
            track_len = struct.unpack('<Q', fid.read(8))[0]
            # Skip track (image_id, point2D_idx) pairs
            fid.seek(track_len * (4 + 4), os.SEEK_CUR)
            points3D[pt_id] = Point3D(id=pt_id, xyz=xyz, rgb=rgb, error=error)
    return points3D

def read_model_binary(sparse_dir: str):
    """Read COLMAP binary model files from *sparse_dir*.
    Returns three dictionaries: cameras, images, points3D.
    """
    cam_path = os.path.join(sparse_dir, 'cameras.bin')
    img_path = os.path.join(sparse_dir, 'images.bin')
    pts_path = os.path.join(sparse_dir, 'points3D.bin')
    if not (os.path.exists(cam_path) and os.path.exists(img_path)):
        raise FileNotFoundError('COLMAP binary model files not found in {}'.format(sparse_dir))
    cameras = _read_cameras_binary(cam_path)
    images = _read_images_binary(img_path)
    points3D = {}
    if os.path.exists(pts_path):
        points3D = _read_points3d_binary(pts_path)
    return cameras, images, points3D

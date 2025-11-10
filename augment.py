import numpy as np

def _split_lr(vec):
    L = np.array(vec[:63], dtype=np.float32).reshape(21,3)
    R = np.array(vec[63:126], dtype=np.float32).reshape(21,3)
    return L, R

def _merge_lr(L, R):
    return np.concatenate([L.reshape(-1), R.reshape(-1)], axis=0).astype(np.float32)

def _rotate_xy(M, deg):
    th = np.deg2rad(float(deg))
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c,-s],[s, c]], dtype=np.float32)
    xy = M[:, :2] @ R.T
    out = M.copy()
    out[:, :2] = xy
    return out

def _scale(M, s):
    out = M.copy()
    out[:, :2] *= float(s)  
    return out

def _jitter(M, std=0.02):
    out = M.copy()
    noise = np.random.normal(0.0, float(std), size=out.shape).astype(np.float32)
    out += noise
    return out

def augment_vec(vec, rot_deg=10, scale_minmax=(0.9,1.1), jitter_std=0.02, num_aug=10):
    """
    Genera num_aug variaciones de un mismo vector de landmarks
    """
    L, R = _split_lr(vec)
    results = []

    for _ in range(num_aug):
        deg = np.random.uniform(-rot_deg, rot_deg)
        sca = np.random.uniform(scale_minmax[0], scale_minmax[1])

        L2 = _jitter(_scale(_rotate_xy(L, deg), sca), jitter_std)
        R2 = _jitter(_scale(_rotate_xy(R, deg), sca), jitter_std)
        results.append(_merge_lr(L2, R2))

    return results

"""
Adapter that converts MovingBarMovieGenerator output to the format expected by
Cricket2RGCs / SynMovieGenerator:
  -> (syn_movie, path, path_bg, *rest)
where syn_movie is a torch.FloatTensor shaped (T, C, H, W),
path and path_bg are numpy arrays shaped (T, 2).
"""

import numpy as np
import torch

class MovingBarAdapter:
    """
    Adapter that converts MovingBarMovieGenerator output to the format expected by
    Cricket2RGCs / SynMovieGenerator:
      -> (syn_movie, path, path_bg, *rest)
    where syn_movie is a torch.FloatTensor shaped (T, C, H, W),
    path and path_bg are numpy arrays shaped (T, 2).
    """
    def __init__(self, moving_bar_generator):
        self.gen = moving_bar_generator

    def generate(self):
        out = self.gen.generate()

        # Case 1: generator returned dict with episodes (test script behavior)
        if isinstance(out, dict) and "episodes" in out:
            if len(out["episodes"]) == 0:
                raise ValueError("MovingBar generator returned empty episodes list")
            ep = out["episodes"][0]
            frames = ep["frames"]          # numpy array expected
            path = ep.get("path", None)
            path_bg = ep.get("path_bg", None)
            rest = (ep.get("meta", None),)
        # Case 2: generator already returns tuple-like (syn_movie, path, path_bg, ...)
        elif isinstance(out, (tuple, list)):
            # keep everything but standardize first three entries
            syn_movie = out[0]
            path = out[1] if len(out) > 1 else None
            path_bg = out[2] if len(out) > 2 else None
            rest = tuple(out[3:]) if len(out) > 3 else ()
            # If syn_movie already in desired torch (T,C,H,W) form, just return
            if isinstance(syn_movie, torch.Tensor) and syn_movie.ndim == 4:
                return (syn_movie, np.asarray(path), np.asarray(path_bg),) + rest
            # Otherwise treat syn_movie as frames like (H,W,T) or (H,W,2,T)
            frames = syn_movie
        else:
            raise TypeError("Unsupported output type from moving bar generator")

        # Convert frames -> torch (T, C, H, W)
        if isinstance(frames, np.ndarray):
            if frames.ndim == 3:
                # (H, W, T) -> (T, 1, H, W)
                H, W, T = frames.shape
                t = torch.from_numpy(frames.astype(np.float32))
                t = t.permute(2, 0, 1).unsqueeze(1)  # (T,1,H,W)
            elif frames.ndim == 4:
                # (H, W, C_or_eyes, T) expected -> (T, C, H, W)
                H, W, C, T = frames.shape
                t = torch.from_numpy(frames.astype(np.float32))
                t = t.permute(3, 2, 0, 1)  # (T,C,H,W)
            else:
                raise ValueError(f"Unexpected frames ndim={frames.ndim}")
        elif isinstance(frames, torch.Tensor):
            # Accept torch arrays but ensure shape is (T,C,H,W)
            if frames.ndim == 3:  # (H,W,T)
                t = frames.permute(2, 0, 1).unsqueeze(1).float()
            elif frames.ndim == 4:
                # could be (T,C,H,W) already or (H,W,C,T)
                if frames.shape[0] < 10 and frames.shape[-1] > 10:
                    # heuristic: (H,W,C,T) -> permute
                    t = frames.permute(3, 2, 0, 1).float()
                else:
                    t = frames.float()
            else:
                raise ValueError("Unsupported torch frames shape")
        else:
            raise TypeError("frames must be numpy.ndarray or torch.Tensor")

        # Ensure values are float32 and in [0,1]
        t = t.float()
        if t.max() > 1.5:  # heuristic: if in [-1,1] scale to [0,1]
            t = (t + 1.0) / 2.0

        # Standardize path and path_bg as numpy arrays with shape (T,2)
        if path is None:
            path_np = np.zeros((t.shape[0], 2), dtype=np.float32)
        else:
            path_np = np.asarray(path, dtype=np.float32)
            if path_np.shape[0] != t.shape[0]:
                # try to align last time-steps (like other generators do)
                if path_np.shape[0] > t.shape[0]:
                    path_np = path_np[-t.shape[0]:]
                else:
                    # pad with repeated last position
                    pad_n = t.shape[0] - path_np.shape[0]
                    last = path_np[-1:].repeat(pad_n, axis=0)
                    path_np = np.vstack((path_np, last))

        if path_bg is None:
            path_bg_np = path_np.copy()
        else:
            path_bg_np = np.asarray(path_bg, dtype=np.float32)
            if path_bg_np.shape[0] != t.shape[0]:
                if path_bg_np.shape[0] > t.shape[0]:
                    path_bg_np = path_bg_np[-t.shape[0]:]
                else:
                    pad_n = t.shape[0] - path_bg_np.shape[0]
                    last = path_bg_np[-1:].repeat(pad_n, axis=0)
                    path_bg_np = np.vstack((path_bg_np, last))

        # Return in the form Cricket2RGCs expects: (syn_movie, path, path_bg, *rest)
        return (t, path_np, path_bg_np) + rest

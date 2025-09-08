import abc
import random
import ast
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Dict, Any, List

from utils.utils import get_random_file_path, get_image_number, get_filename_without_extension, load_mat_to_dataframe
from utils.trajectory import disparity_from_scaling_factor, convert_deg_to_pix, adjust_trajectories
from datasets.sim_cricket import random_movement, calculate_scaling_factors, synthesize_image_with_params_batch  # reuse existing helpers

class BaseMovieGenerator(abc.ABC):
    """
    Base class that encapsulates shared configuration and helpers for movie generators.
    Subclasses implement generate() for specific stimulus types (cricket, bg-only, moving bar, checkerboard, ...).
    """
    def __init__(
        self,
        crop_size: Tuple[int, int],
        boundary_size,
        center_ratio=(0.2, 0.2),
        max_steps: int = 200,
        num_ext: int = 50,
        is_binocular: bool = False,
        interocular_dist: float = 1.0,
        coord_mat_file: Optional[str] = None,
        correction_direction: int = 1,
        is_reverse_xy: bool = False,
        bottom_contrast: float = 1.0,
        top_contrast: float = 1.0,
        mean_diff_offset: float = 0.0,
        fix_disparity: Optional[float] = None,
        **kwargs
    ):
        self.crop_size = crop_size
        self.boundary_size = tuple(ast.literal_eval(boundary_size)) if isinstance(boundary_size, str) else tuple(boundary_size)
        self.center_ratio = center_ratio
        self.max_steps = max_steps
        self.num_ext = num_ext
        self.is_binocular = is_binocular
        self.interocular_dist = interocular_dist
        self.coord_dic = self._get_coord_dic(coord_mat_file)
        self.correction_direction = correction_direction
        self.is_reverse_xy = is_reverse_xy
        self.bottom_contrast = bottom_contrast
        self.top_contrast = top_contrast
        self.mean_diff_offset = mean_diff_offset
        self.fix_disparity = fix_disparity

        # store any other extra params for subclasses
        self.extra_params = kwargs

    # --- Core API for subclasses ---
    @abc.abstractmethod
    def generate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Must return (syn_movie, path, path_bg, meta)
        - syn_movie: tensor/ndarray with frames (support mono or binocular)
        - path: object positions (T,2)
        - path_bg: background positions (T,2)
        - meta: dict with extra outputs (scaling_factors, bg_image_name, image_id, ...)
        """
        raise NotImplementedError

    # --- Shared helpers (override in subclasses when needed) ---
    def _get_coord_dic(self, coord_mat_file):
        if coord_mat_file is None:
            return None
        df = load_mat_to_dataframe(coord_mat_file)
        index_column_name = 'image_id'
        if index_column_name not in df.columns:
            raise ValueError(f"Expected '{index_column_name}' column in coord mat file")
        return df.set_index(index_column_name).to_dict(orient='index')

    def _modify_scaling(self, start_scaling: float, end_scaling: float, dynamic_scaling: float = 0.0):
        s, e = start_scaling, end_scaling
        if dynamic_scaling != 0:
            s += random.uniform(0, dynamic_scaling)
            e -= random.uniform(0, dynamic_scaling)
            s, e = min(s, e), max(s, e)
        return s, e

    def _sample_paths(self, prob_stay_ob, prob_mov_ob, prob_stay_bg, prob_mov_bg,
                      initial_velocity=1.0, momentum_decay_ob=0.95, momentum_decay_bg=0.5,
                      velocity_randomness_ob=0.02, velocity_randomness_bg=0.01,
                      angle_range_ob=0.5, angle_range_bg=0.25):
        path, velocity = random_movement(self.boundary_size, self.center_ratio, self.max_steps,
                                         prob_stay=prob_stay_ob, prob_mov=prob_mov_ob,
                                         initial_velocity=initial_velocity, momentum_decay=momentum_decay_ob,
                                         velocity_randomness=velocity_randomness_ob, angle_range=angle_range_ob)
        path_bg, velocity_bg = random_movement(self.boundary_size, self.center_ratio, self.max_steps,
                                               prob_stay=prob_stay_bg, prob_mov=prob_mov_bg,
                                               initial_velocity=initial_velocity, momentum_decay=momentum_decay_bg,
                                               velocity_randomness=velocity_randomness_bg, angle_range=angle_range_bg)
        min_len = min(len(path), len(path_bg))
        path, velocity, path_bg, velocity_bg = (path[:min_len], velocity[:min_len], path_bg[:min_len], velocity_bg[:min_len])
        # extend static frames at start
        path = np.vstack((np.repeat(path[0:1, :], self.num_ext, axis=0), path))
        path_bg = np.vstack((np.repeat(path_bg[0:1, :], self.num_ext, axis=0), path_bg))
        return path, path_bg, velocity, velocity_bg

    def _apply_binocular(self, path, scaling_factors, start_distance=21, end_distance=4):
        """Return averaged/adjusted path and shifted positions for binocular rendering."""
        disparity, _ = disparity_from_scaling_factor(
            scaling_factors=scaling_factors,
            start_distance=start_distance,
            end_distance=end_distance,
            iod_cm=self.interocular_dist,
            fix_disparity=self.fix_disparity
        )
        disparity = convert_deg_to_pix(disparity)
        top_img_disparity_positions = path.copy()
        top_img_disparity_positions[:, 0] += disparity
        bounds = (-self.boundary_size[0]/2, self.boundary_size[0]/2, -self.boundary_size[1]/2, self.boundary_size[1]/2)
        top_shifted, top_disp_shifted = adjust_trajectories(bounds, path.copy(), top_img_disparity_positions)
        avg_path = (top_shifted + top_disp_shifted) / 2.0
        return avg_path, top_shifted, top_disp_shifted

    def _load_random_pair(self, top_folder: Optional[str], bottom_folder: Optional[str]):
        """Return (bottom_img_path, top_img_path) — allow None for top when bg-only generator."""
        bottom = get_random_file_path(bottom_folder) if bottom_folder is not None else None
        top = get_random_file_path(top_folder) if top_folder is not None else None
        return bottom, top

    def _correct_coord(self, path, image_id, scaling_factors):
        if self.coord_dic is None:
            return path
        if self.is_reverse_xy:
            coord = np.array([self.coord_dic[image_id]['coord_y'], self.coord_dic[image_id]['coord_x']])
        else:
            coord = np.array([self.coord_dic[image_id]['coord_x'], self.coord_dic[image_id]['coord_y']])
        scaled = coord[np.newaxis, :] * np.array(scaling_factors)[:, np.newaxis]
        return path - self.correction_direction * scaled

    def _synthesize_batch(self, bottom_img_path, top_img_path, top_positions, bottom_positions,
                          scaling_factors, alpha=1.0, top_shifted_positions=None, **synth_kwargs):
        """
        Default batch synthesizer — subclasses may override to produce bars, checkerboards, or other stimuli.
        Uses existing synthesize_image_with_params_batch when top_img_path is not None.
        """
        if top_img_path is None:
            # background-only: return repeated/cropped background frames (subclass can override)
            # default: call synthesize with top_img_path=None -> implementers should accept this
            return synthesize_image_with_params_batch(bottom_img_path, None, top_positions, bottom_positions,
                                                      scaling_factors, self.crop_size, alpha=alpha,
                                                      top_img_positions_shifted=top_shifted_positions,
                                                      bottom_contrast=self.bottom_contrast,
                                                      top_contrast=self.top_contrast,
                                                      mean_diff_offset=self.mean_diff_offset)
        return synthesize_image_with_params_batch(bottom_img_path, top_img_path, top_positions, bottom_positions,
                                                 scaling_factors, self.crop_size, alpha=alpha,
                                                 top_img_positions_shifted=top_shifted_positions,
                                                 bottom_contrast=self.bottom_contrast,
                                                 top_contrast=self.top_contrast,
                                                 mean_diff_offset=self.mean_diff_offset)

    # --- Small utility ---
    def _get_bg_name_and_top_id(self, bottom_img_path, top_img_path):
        bg_image_name = get_filename_without_extension(bottom_img_path) if bottom_img_path is not None else ""
        image_id = get_image_number(top_img_path) if top_img_path is not None else None
        return bg_image_name, image_id

    # Hook for subclasses to modify frames (e.g. contrast, mean)
    def postprocess_movie(self, syn_movie):
        return syn_movie
    

class MovingBarMovieGenerator(BaseMovieGenerator):
    """
    Generate episodes of a moving bar stimulus.
    Each episode:
      - random bar width and height (pixels) from provided ranges
      - random speed (pixels/frame) from provided range
      - random movement direction (degrees) from provided range
      - movement line always crosses screen center
      - start/end are placed so the bar is fully off-screen (no bar visible at first/last frame)
      - contrast c in [-1,1] maps to bar = 0.5 + c/2, bg = 0.5 - c/2
    """
    def __init__(
        self,
        crop_size: Tuple[int, int],
        boundary_size,
        bar_width_range: Tuple[int, int] = (10, 40),
        bar_height_range: Tuple[int, int] = (60, 300),
        speed_range: Tuple[float, float] = (4.0, 12.0),
        direction_range: Tuple[float, float] = (0.0, 360.0),  # degrees for movement direction
        num_episodes: int = 1,
        margin: float = 2.0,
        disparity_range: Tuple[float, float] = (0.5, 4.0),  # disparity range in pixels for binocular
        **kwargs
    ):
        super().__init__(crop_size=crop_size, boundary_size=boundary_size, **kwargs)
        self.bar_width_range = bar_width_range
        self.bar_height_range = bar_height_range
        self.speed_range = speed_range
        self.direction_range = direction_range
        self.num_episodes = num_episodes
        self.margin = margin
        self.disparity_range = disparity_range

    def _make_bar_mask(self, W: int, H: int, center_xy: Tuple[float, float], width: float, height: float, angle_deg: float):
        """
        Return boolean mask (H, W) where rotated rectangle centered at center_xy with given width (along movement axis)
        and height (perp to movement) is True. angle_deg is bar orientation in degrees (0 => x-axis).
        """
        cy, cx = center_xy[1], center_xy[0]
        # Create coordinate grids: xx has shape (H, W) with x-coordinates, yy has shape (H, W) with y-coordinates
        xx, yy = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
        # coordinates relative to bar center
        xr = xx - center_xy[0]
        yr = yy - center_xy[1]
        theta = np.deg2rad(angle_deg)
        # rotate coordinates so bar aligned with axes
        cos_t = np.cos(-theta)
        sin_t = np.sin(-theta)
        xrot = xr * cos_t - yr * sin_t
        yrot = xr * sin_t + yr * cos_t
        mask = (np.abs(xrot) <= width/2.0) & (np.abs(yrot) <= height/2.0)
        return mask

    def _compute_start_distance(self, W:int, H:int, bar_w:float, bar_h:float):
        # radius from center to the farthest corner of the screen
        screen_radius = np.sqrt((W/2.0)**2 + (H/2.0)**2)
        bar_half_diag = np.sqrt((bar_w/2.0)**2 + (bar_h/2.0)**2)
        return screen_radius + bar_half_diag + self.margin

    def _contrast_colors(self, c: float):
        # c in [-1,1]. bar = 0.5 + c/2 ; bg = 0.5 - c/2
        c = float(np.clip(c, -1.0, 1.0))
        bar = 0.5 + c/2.0
        bg = 0.5 - c/2.0
        return bar, bg

    def _compute_disparity(self, n_frames):
        """
        Compute disparity values for binocular rendering.
        For moving bars, disparity can be constant or vary over time.
        """
        if self.fix_disparity is not None:
            # Fixed disparity case
            disparity = np.full(n_frames, self.fix_disparity, dtype=np.float32)
        else:
            # Random disparity from the specified range
            # For moving bars, we'll use constant disparity per episode
            base_disparity = np.random.uniform(self.disparity_range[0], self.disparity_range[1])
            disparity = np.full(n_frames, base_disparity, dtype=np.float32)
        
        return disparity

    def _make_episode(self):
        W, H = int(self.crop_size[0]), int(self.crop_size[1])
        # randomize parameters
        bar_w = float(np.random.randint(self.bar_width_range[0], self.bar_width_range[1]+1))
        bar_h = float(np.random.randint(self.bar_height_range[0], self.bar_height_range[1]+1))
        speed = float(np.random.uniform(self.speed_range[0], self.speed_range[1]))
        move_dir = float(np.random.uniform(self.direction_range[0], self.direction_range[1]))  # degrees
        # movement direction is along move_dir (angle). bar orientation is perpendicular to movement
        bar_angle = (move_dir + 90.0) % 360.0
        # compute start/end distances along movement axis so bar fully off-screen
        start_dist = self._compute_start_distance(W, H, bar_w, bar_h)
        total_travel = 2.0 * start_dist
        n_frames = max(2, int(np.ceil(total_travel / speed)) + 1)
        
        # movement and positioning
        dx = np.cos(np.deg2rad(move_dir))
        dy = np.sin(np.deg2rad(move_dir))
        center_screen = np.array([W/2.0, H/2.0])
        
        # contrast
        contrast = float(np.random.uniform(-1.0, 1.0))
        bar_col, bg_col = self._contrast_colors(contrast)
        
        # generate center positions (without disparity)
        ts = np.linspace(-start_dist, start_dist, n_frames)
        center_positions = []
        for t in ts:
            cen = center_screen + np.array([dx * t, dy * t])
            center_positions.append(cen.copy())
        center_positions = np.vstack(center_positions)  # (T, 2)
        
        if self.is_binocular:
            # Compute disparity for binocular rendering
            disparity = self._compute_disparity(n_frames)
            
            # Create left and right eye positions
            # Left eye: shift by -disparity/2 in x, right eye: shift by +disparity/2 in x
            left_positions = center_positions.copy()
            right_positions = center_positions.copy()
            left_positions[:, 0] -= disparity / 2.0   # shift left
            right_positions[:, 0] += disparity / 2.0  # shift right
            
            # Generate frames for both eyes
            frames = np.zeros((H, W, 2, n_frames), dtype=np.float32)  # (H, W, 2_eyes, T)
            
            # Left eye frames (eye index 0)
            for i in range(n_frames):
                mask = self._make_bar_mask(W, H, (left_positions[i, 0], left_positions[i, 1]), 
                                         width=bar_w, height=bar_h, angle_deg=bar_angle)
                frames[:, :, 0, i] = bg_col
                frames[:, :, 0, i][mask] = bar_col
            
            # Right eye frames (eye index 1)  
            for i in range(n_frames):
                mask = self._make_bar_mask(W, H, (right_positions[i, 0], right_positions[i, 1]), 
                                         width=bar_w, height=bar_h, angle_deg=bar_angle)
                frames[:, :, 1, i] = bg_col
                frames[:, :, 1, i][mask] = bar_col
            
            # For path output, use the average (center) positions
            path = center_positions
            
            # Add binocular info to metadata
            meta = dict(
                bar_width=bar_w, bar_height=bar_h, speed=speed, move_dir=move_dir,
                bar_angle=bar_angle, contrast=contrast, n_frames=n_frames,
                start_dist=start_dist, center=(W/2.0, H/2.0),
                is_binocular=True, disparity=disparity,
                left_positions=left_positions, right_positions=right_positions
            )
            
        else:
            # Monocular case (original implementation)
            frames = np.zeros((H, W, n_frames), dtype=np.float32)
            
            for i in range(n_frames):
                mask = self._make_bar_mask(W, H, (center_positions[i, 0], center_positions[i, 1]), 
                                         width=bar_w, height=bar_h, angle_deg=bar_angle)
                frames[:, :, i] = bg_col
                frames[:, :, i][mask] = bar_col
            
            path = center_positions
            meta = dict(
                bar_width=bar_w, bar_height=bar_h, speed=speed, move_dir=move_dir,
                bar_angle=bar_angle, contrast=contrast, n_frames=n_frames,
                start_dist=start_dist, center=(W/2.0, H/2.0),
                is_binocular=False
            )
        
        path_bg = None  # No background movement for moving bars
        return frames, path, path_bg, meta

    def generate(self, num_episodes: int = None) -> Dict[str, Any]:
        """
        Generate num_episodes (default self.num_episodes). Returns dict:
          {
            "episodes": [
               {"frames": np.ndarray(H,W,T), "path": (T,2), "path_bg": None, "meta": {...}},
               ...
            ]
          }
        """
        if num_episodes is None:
            num_episodes = self.num_episodes
        episodes = []
        for _ in range(int(num_episodes)):
            frames, path, path_bg, meta = self._make_episode()
            episodes.append({"frames": frames, "path": path, "path_bg": path_bg, "meta": meta})
        return {"episodes": episodes}
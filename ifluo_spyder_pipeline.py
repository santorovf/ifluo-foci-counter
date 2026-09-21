"""
Immunofluorescence Analysis (Spyder friendly, single file)
----------------------------------------------------------
- Input: path to a single image OR a folder of images (RGB). B = cells, G = foci.
- Output: overlay PNGs + per cell CSV per image.
- Usage: set INPUT_PATH and OUTPUT_DIR, then run (F5).

New features:
1. Control based thresholding, use CONTROL or CTL filenames as baselines.
2. Robust seed cap, prevent excessive thresholds when noise inflates SD.
3. Seeds overlay image, saves an image with tiny "x" at seed peaks (pre watershed).
4. Seed inclusion in growth mask, guarantees markers can expand.
5. USE_CONTROLS switch, set False to threshold every image on its own statistics.
6. CONTROL_PERCENTILE / DETECTION_PERCENTILE settings (default 99.5).
"""

import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, filters, morphology, measure, segmentation
from scipy import ndimage as ndi

# -----------------------------
# USER SETTINGS
# -----------------------------
INPUT_PATH  = r"C:\path\to\your\images"   # a folder of images OR a single image file
OUTPUT_DIR  = r"C:\path\to\output"
EXCLUDE_SCALE_BAR = True
EXCLUDE_FRACTION  = (0.10, 0.30)
USE_CONTROLS = True        # False = ignore CONTROL/CTRL/CTL files, threshold every image on its own stats
CONTROL_PERCENTILE = 99.5  # percentile of control cell thresholds used as the seed/growth floor
DETECTION_PERCENTILE = 99.5  # percentile of cell pixels that caps the seed threshold (robust seed cap)

# -----------------------------
# Helper utilities
# -----------------------------
def _to_float01(img):
    imgf = img.astype(np.float32)
    if imgf.max() > 1.0:
        imgf = imgf / 255.0
    return imgf

def is_control_name(path):
    name = os.path.basename(path).lower()
    return re.search(r'(control|ctrl|ctl)', name) is not None

# -----------------------------
# Core segmentation
# -----------------------------
def build_coarse_mask(imgf, sigma=1.0, otsu_scale=0.6, min_obj=100, close_radius=3,
                      exclude_scale_bar=True, exclude_fraction=(0.10, 0.30)):
    B = imgf[..., 2]
    G = imgf[..., 1]
    H, W = B.shape
    fusion = 0.5 * B + 0.5 * G
    fusion_smooth = filters.gaussian(fusion, sigma=sigma, preserve_range=True)
    thr_otsu = filters.threshold_otsu(fusion_smooth)
    thr = thr_otsu * otsu_scale
    coarse = fusion_smooth > thr
    if exclude_scale_bar:
        h_start = int(H * (1.0 - exclude_fraction[0]))
        w_start = int(W * (1.0 - exclude_fraction[1]))
        coarse[h_start:, w_start:] = False
    coarse = morphology.remove_small_objects(coarse, min_obj)
    coarse = ndi.binary_fill_holes(coarse)
    coarse = morphology.binary_closing(coarse, morphology.disk(close_radius))
    coarse = ndi.binary_fill_holes(coarse)
    return coarse

def refine_cells_local_bg(imgf, coarse_mask, pad=6, base_factor=0.6,
                          raise_coeff_sigmaB=0.25, min_part_area=250):
    B = imgf[..., 2]
    G = imgf[..., 1]
    H, W = B.shape
    fusion = 0.5 * B + 0.5 * G
    fusion_smooth = filters.gaussian(fusion, sigma=1.0, preserve_range=True)
    labels_coarse = measure.label(coarse_mask, connectivity=1)
    bg_vals = fusion_smooth[~coarse_mask]
    global_bg_mean = float(bg_vals.mean()) if bg_vals.size else float(fusion_smooth.mean())
    labels_refined = np.zeros_like(labels_coarse, dtype=np.int32)
    next_id = 1

    for region in measure.regionprops(labels_coarse):
        minr, minc, maxr, maxc = region.bbox
        minr_p = max(minr - pad, 0); minc_p = max(minc - pad, 0)
        maxr_p = min(maxr + pad, H); maxc_p = min(maxc + pad, W)
        reg_full = labels_coarse[minr_p:maxr_p, minc_p:maxc_p] == region.label
        loc_fusion = fusion[minr_p:maxr_p, minc_p:maxc_p]
        iso = np.full_like(loc_fusion, global_bg_mean, dtype=loc_fusion.dtype)
        iso[reg_full] = loc_fusion[reg_full]
        loc_B = B[minr_p:maxr_p, minc_p:maxc_p]
        sigma_B = float(np.std(loc_B[reg_full])) if np.any(reg_full) else 0.0
        iso_s = filters.gaussian(iso, sigma=1.0, preserve_range=True)
        otsu_local = filters.threshold_otsu(iso_s)
        thr_local = otsu_local * base_factor + raise_coeff_sigmaB * sigma_B
        refined_mask = iso_s > thr_local
        refined_mask = morphology.remove_small_objects(refined_mask, 100)
        refined_mask = ndi.binary_fill_holes(refined_mask)
        refined_mask = morphology.binary_closing(refined_mask, morphology.disk(2))
        parts = measure.label(refined_mask, connectivity=1)
        kept_parts = []
        for pid in range(1, parts.max() + 1):
            part = (parts == pid)
            if part.sum() >= min_part_area and np.any(part & reg_full):
                kept_parts.append(part)
        if len(kept_parts) >= 2:
            for part in kept_parts:
                labels_refined[minr_p:maxr_p, minc_p:maxc_p][part] = next_id
                next_id += 1
        else:
            labels_refined[minr:maxr, minc:maxc][labels_coarse[minr:maxr, minc:maxc] == region.label] = next_id
            next_id += 1

    # build filled mask per cell
    labels_final = np.zeros_like(labels_refined, dtype=np.int32)
    new_id = 1
    for lab in range(1, labels_refined.max() + 1):
        m = labels_refined == lab
        if not m.any():
            continue
        m_filled = ndi.binary_fill_holes(m)
        labels_final[m_filled] = new_id
        new_id += 1

    # remove cells that touch the image border
    Hf, Wf = labels_final.shape
    border_mask = np.zeros_like(labels_final, dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True
    border_labels = np.unique(labels_final[border_mask])
    for lab in border_labels:
        if lab == 0:
            continue
        labels_final[labels_final == lab] = 0

    # remove cells smaller than 250 pixels
    MIN_CELL_SIZE = 250
    for r in measure.regionprops(labels_final):
        if r.area < MIN_CELL_SIZE:
            labels_final[labels_final == r.label] = 0

    # relabel remaining cells to consecutive ids
    labels_final, _, _ = segmentation.relabel_sequential(labels_final)

    return labels_final

# -----------------------------
# Threshold estimator (control based)
# -----------------------------
def estimate_floors_from_controls(paths, exclude_scale_bar=True, exclude_fraction=(0.10, 0.30),
                                  control_percentile=99.5, fallback_seed=0.1, fallback_growth=0.07):
    control_paths = [p for p in paths if is_control_name(p)]
    print(f"[Pre run] Found {len(control_paths)} control images.")
    if control_paths:
        for c in control_paths:
            print("   control:", os.path.basename(c))
    if not control_paths:
        return None, None

    t_seed_all, t_growth_all = [], []
    for p in control_paths:
        try:
            imgf = _to_float01(io.imread(p))
            G = imgf[..., 1]
            coarse = build_coarse_mask(imgf, exclude_scale_bar=exclude_scale_bar, exclude_fraction=exclude_fraction)
            if not np.any(coarse):
                continue
            labels = refine_cells_local_bg(imgf, coarse)
            for r in measure.regionprops(labels):
                mask = labels == r.label
                vals = G[mask]
                if vals.size == 0:
                    continue
                mu, sd = float(vals.mean()), float(vals.std())
                t_seed_all.append(mu + 3*sd)
                t_growth_all.append(mu + 1.5*sd)
        except Exception as e:
            print(f"Skipping {os.path.basename(p)}: {e}")

    if len(t_seed_all) == 0:
        return fallback_seed, fallback_growth

    g_seed = np.percentile(t_seed_all, control_percentile)
    g_growth = np.percentile(t_growth_all, control_percentile)
    g_seed = float(np.clip(g_seed, 0, 1))
    g_growth = float(np.clip(g_growth, 0, 1))
    print(f"[Pre run] Control based floors → seed={g_seed:.4f}, growth={g_growth:.4f}")
    return g_seed, g_growth

# -----------------------------
# Save seed overlay
# -----------------------------
def save_seed_overlay(base_rgb_uint8, labels_cells, seed_points_xy, out_path):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(base_rgb_uint8)
    ax.axis('off')
    for r in measure.regionprops(labels_cells):
        m = labels_cells == r.label
        for c in measure.find_contours(m, 0.5):
            ax.plot(c[:,1], c[:,0], color='red', linewidth=0.8)
        cy, cx = r.centroid
        ax.text(cx, cy, str(r.label), color='yellow', fontsize=8, ha='center', va='center')
    if seed_points_xy:
        xs, ys = zip(*seed_points_xy)
        ax.plot(xs, ys, linestyle='None', marker='x', markersize=4, markeredgewidth=0.8, color='yellow')
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

# -----------------------------
# Foci detection (with overlay)
# -----------------------------
def detect_foci_seeded_locked(G, labels_cells, global_min_seed=0.0, global_min_growth=0.0,
                              robust_quantile=0.995, debug=False,
                              seed_overlay_path=None, base_rgb_uint8=None):

    MIN_CELL_SIZE = 250
    MIN_FOCI_SIZE = 5

    H, W = G.shape
    foci_global = np.zeros((H, W), dtype=np.int32)
    next_id = 1
    records = []
    seed_points_xy = []

    # global spot image for watershed
    spot_global = morphology.white_tophat(G, footprint=morphology.disk(3))
    spot_global = filters.gaussian(spot_global, sigma=0.8, preserve_range=True)

    # global mean and std over all cell pixels
    all_cell_mask = labels_cells > 0
    all_vals = G[all_cell_mask]
    if all_vals.size == 0:
        if debug:
            print("[detect_foci_seeded_locked] No cell pixels found, returning empty result.")
        if seed_overlay_path and base_rgb_uint8 is not None:
            save_seed_overlay(base_rgb_uint8, labels_cells, seed_points_xy, seed_overlay_path)
        return foci_global, records

    mu_all = float(all_vals.mean())
    sd_all = float(all_vals.std())
    q_robust_all = float(np.quantile(all_vals, robust_quantile))

    seed_thr_global = min(max(mu_all + 3 * sd_all, global_min_seed), q_robust_all)
    growth_thr_global = max(mu_all + 1.5 * sd_all, global_min_growth)

    # NEW: first pass, compute per cell green intensity per area
    gpa_list = []   # green per area
    for r in measure.regionprops(labels_cells):
        lab = r.label
        cell_mask = (labels_cells == lab)
        cell_area = int(cell_mask.sum())

        # only consider cells that would be eligible for counting
        if cell_area < MIN_CELL_SIZE:
            continue

        g_vals = G[cell_mask]
        if g_vals.size == 0:
            continue

        green_sum = float(g_vals.sum())
        green_per_area = green_sum / cell_area
        gpa_list.append(green_per_area)

    if len(gpa_list) == 0:
        if debug:
            print("[detect_foci_seeded_locked] No valid cells (area >= MIN_CELL_SIZE) for GPA stats.")
        if seed_overlay_path and base_rgb_uint8 is not None:
            save_seed_overlay(base_rgb_uint8, labels_cells, seed_points_xy, seed_overlay_path)
        return foci_global, records

    gpa_array = np.array(gpa_list, dtype=np.float32)
    mean_gpa = float(gpa_array.mean())
    std_gpa = float(gpa_array.std())
    gpa_thr = mean_gpa + 3.0 * std_gpa

    if debug:
        print(f"[global] mu_all={mu_all:.3f}, sd_all={sd_all:.3f}, "
              f"seed_thr_global={seed_thr_global:.3f}, growth_thr_global={growth_thr_global:.3f}")
        print(f"[global GPA] mean={mean_gpa:.3f}, std={std_gpa:.3f}, thr={gpa_thr:.3f}")

    # second pass, cell by cell, but thresholds are global
    for r in measure.regionprops(labels_cells):
        lab = r.label
        minr, minc, maxr, maxc = r.bbox
        cell_mask = labels_cells[minr:maxr, minc:maxc] == lab
        cell_area = int(cell_mask.sum())

        # skip cells smaller than minimum size
        if cell_area < MIN_CELL_SIZE:
            continue

        g_patch = G[minr:maxr, minc:maxc]
        vals_cell = g_patch[cell_mask]
        if vals_cell.size == 0:
            continue

        # compute this cell's green intensity per area
        green_sum_cell = float(vals_cell.sum())
        green_per_area_cell = green_sum_cell / cell_area

        # NEW: skip "hot" cells with green/area > mean + 3*std
        if green_per_area_cell > gpa_thr:
            if debug:
                print(f"[cell {lab}] skipped as HOT, GPA={green_per_area_cell:.3f} > {gpa_thr:.3f}")
            continue

        spot = spot_global[minr:maxr, minc:maxc]

        seed_thr = seed_thr_global
        growth_thr = growth_thr_global

        seed_mask = (g_patch > seed_thr) & cell_mask
        growth_mask = ((g_patch > growth_thr) | seed_mask) & cell_mask

        seed_labels = measure.label(seed_mask, connectivity=1)

        for s in measure.regionprops(seed_labels):
            cy, cx = s.centroid
            seed_points_xy.append((minc + cx, minr + cy))

        if debug:
            print(f"[cell {lab}] area={cell_area}, GPA={green_per_area_cell:.3f}, "
                  f"seeds={seed_labels.max()}")

        labels_ws = segmentation.watershed(-spot, markers=seed_labels, mask=growth_mask)

        # enforce minimum foci size
        nf = 0
        for k in range(1, int(labels_ws.max()) + 1):
            region_mask = (labels_ws == k)
            if region_mask.sum() < MIN_FOCI_SIZE:
                labels_ws[region_mask] = 0
            else:
                nf += 1

        # assign final foci ids in global foci mask
        for k in range(1, int(labels_ws.max()) + 1):
            if np.any(labels_ws == k):
                foci_global[minr:maxr, minc:maxc][labels_ws == k] = next_id
                next_id += 1

        foci_area = int((labels_ws > 0).sum())
        foci_ratio = (foci_area / cell_area) if cell_area else 0
        foci_sum = float(g_patch[labels_ws > 0].sum())

        records.append(dict(
            cell_id=int(lab),
            cell_area_px=cell_area,
            n_foci=nf,
            foci_area_px=foci_area,
            foci_area_ratio=foci_ratio,
            foci_intensity_sum=foci_sum
        ))

    if seed_overlay_path and base_rgb_uint8 is not None:
        save_seed_overlay(base_rgb_uint8, labels_cells, seed_points_xy, seed_overlay_path)

    return foci_global, records

# -----------------------------
# Visualization
# -----------------------------
def overlay_cells_and_foci(img_uint8, labels_cells, labels_foci, show_ids=True):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(img_uint8)
    ax.axis('off')
    for r in measure.regionprops(labels_cells):
        m = labels_cells == r.label
        for c in measure.find_contours(m, 0.5):
            ax.plot(c[:,1], c[:,0], color='red', linewidth=0.8)
        if show_ids:
            cy, cx = r.centroid
            ax.text(cx, cy, str(r.label), color='yellow', fontsize=8, ha='center', va='center')
    for k in range(1, int(labels_foci.max())+1):
        m = labels_foci == k
        for c in measure.find_contours(m, 0.5):
            ax.plot(c[:,1], c[:,0], color='magenta', linewidth=0.8)
    fig.tight_layout()
    return fig, ax

# -----------------------------
# Processing wrapper
# -----------------------------
def process_image(path, out_dir, exclude_scale_bar=True, exclude_fraction=(0.10,0.30),
                  global_min_seed=0.0, global_min_growth=0.0, detection_percentile=99.5):
    os.makedirs(out_dir, exist_ok=True)
    imgf = _to_float01(io.imread(path))
    coarse = build_coarse_mask(imgf, exclude_scale_bar=exclude_scale_bar, exclude_fraction=exclude_fraction)
    labels_cells = refine_cells_local_bg(imgf, coarse)
    G = imgf[...,1]
    name = os.path.splitext(os.path.basename(path))[0]
    seed_overlay_path = os.path.join(out_dir, f"{name}_seeds_overlay.png")

    labels_foci, recs = detect_foci_seeded_locked(G, labels_cells,
                                                  global_min_seed, global_min_growth,
                                                  robust_quantile=detection_percentile / 100.0,
                                                  seed_overlay_path=seed_overlay_path,
                                                  base_rgb_uint8=(imgf*255).astype(np.uint8))
    fig, ax = overlay_cells_and_foci((imgf*255).astype(np.uint8), labels_cells, labels_foci)
    out_img = os.path.join(out_dir, f"{name}_overlay.png")
    fig.savefig(out_img, dpi=300, bbox_inches='tight'); plt.close(fig)
    df = pd.DataFrame(recs); df.to_csv(os.path.join(out_dir, f"{name}_per_cell.csv"), index=False)
    return out_img, len(df), int(df["n_foci"].sum()) if "n_foci" in df else 0

def process_input_path(input_path, output_dir, exclude_scale_bar=True, exclude_fraction=(0.10,0.30),
                       use_controls=True, control_percentile=99.5, detection_percentile=99.5):
    from glob import glob
    if os.path.isdir(input_path):
        paths = sorted([p for p in glob(os.path.join(input_path,"*")) if os.path.isfile(p)])
        if use_controls:
            g_seed, g_growth = estimate_floors_from_controls(paths, exclude_scale_bar=exclude_scale_bar,
                                                             exclude_fraction=exclude_fraction,
                                                             control_percentile=control_percentile)
            if g_seed is None:
                g_seed, g_growth = 0.05, 0.03
        else:
            print("[Pre run] USE_CONTROLS = False, thresholds come from each image's own statistics.")
            g_seed, g_growth = 0.0, 0.0
    else:
        paths = [input_path]
        g_seed, g_growth = 0.0, 0.0
    print(f"[Run] Using floors seed={g_seed:.4f}, growth={g_growth:.4f}, "
          f"detection percentile={detection_percentile}")
    os.makedirs(output_dir, exist_ok=True)
    summary = []
    for p in paths:
        try:
            out_dir_i = os.path.join(output_dir, os.path.splitext(os.path.basename(p))[0])
            img_path, n_cells, n_foci = process_image(p, out_dir_i,
                                                     exclude_scale_bar, exclude_fraction,
                                                     g_seed, g_growth, detection_percentile)
            summary.append(dict(image=p, cells=n_cells, foci_total=n_foci))
            print(f"[OK] {os.path.basename(p)} → cells={n_cells}, foci={n_foci}")
        except Exception as e:
            print(f"[ERROR] {p}: {e}")
    pd.DataFrame(summary).to_csv(os.path.join(output_dir,"summary.csv"), index=False)

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"INPUT_PATH not found: {INPUT_PATH}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name, pct in (("CONTROL_PERCENTILE", CONTROL_PERCENTILE), ("DETECTION_PERCENTILE", DETECTION_PERCENTILE)):
        if not 0 < pct <= 100:
            raise ValueError(f"{name} must be in (0, 100], got {pct}")
    process_input_path(INPUT_PATH, OUTPUT_DIR, exclude_scale_bar=EXCLUDE_SCALE_BAR, exclude_fraction=EXCLUDE_FRACTION,
                       use_controls=USE_CONTROLS, control_percentile=CONTROL_PERCENTILE,
                       detection_percentile=DETECTION_PERCENTILE)
    print("Done.")

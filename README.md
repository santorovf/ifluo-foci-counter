# IF foci counter

Counts nuclear foci (for example γH2AX) in immunofluorescence images, cell by cell.
Cells are segmented from the **blue** channel and foci are detected in the **green** channel.

For each image it saves an overlay PNG, a seed overlay PNG and a per-cell CSV. It also writes a `summary.csv` for the whole run.

## Setup (once)

1. Install Python 3.10 or newer, or use Anaconda/Spyder.
2. Download this repository: **Code → Download ZIP**, or `git clone`.
3. Install the dependencies:

   ```
   pip install -r requirements.txt
   ```

   In Anaconda, run this from the Anaconda Prompt.

## Running

1. Open `ifluo_spyder_pipeline.py` in Spyder (or any editor).
2. Edit the **USER SETTINGS** block at the top:

   | Setting | What it does | Default |
   |---|---|---|
   | `INPUT_PATH` | A folder of RGB images, or a single image file | – |
   | `OUTPUT_DIR` | Where results are written | – |
   | `EXCLUDE_SCALE_BAR` | Masks the bottom-right corner so the scale bar is not counted as a cell | `True` |
   | `EXCLUDE_FRACTION` | Size of that masked corner (height, width) as a fraction of the image | `(0.10, 0.30)` |
   | `USE_CONTROLS` | Use control images to set a minimum threshold (see below) | `True` |
   | `CONTROL_PERCENTILE` | Percentile of control cell thresholds used as that minimum | `99.5` |
   | `DETECTION_PERCENTILE` | Percentile of cell pixel intensities that caps the seed threshold in every image | `99.5` |

3. Run the script (F5 in Spyder), or from a terminal: `python ifluo_spyder_pipeline.py`.

## How the foci threshold is set

In each image, the foci threshold is based on the green intensity of all cell pixels in that image. Its mean is μ and its standard deviation is σ.

- **Seed threshold:** `min(max(μ + 3σ, control floor), DETECTION_PERCENTILE of cell pixels)`
- **Growth threshold:** `max(μ + 1.5σ, control floor)`

Seeds are pixels above the seed threshold. They are grown by watershed into pixels above the growth threshold. Foci smaller than 5 px are dropped. Cells smaller than 250 px are skipped, and so are "hot" cells (green per area > mean + 3 SD across cells).

### With controls (`USE_CONTROLS = True`)

Any file whose name contains `control`, `ctrl` or `ctl` (not case-sensitive) is treated as a control. For each control cell, μ+3σ and μ+1.5σ are computed. The `CONTROL_PERCENTILE` of those values becomes the floor, so no image can use a threshold below that level.

Watch file names: `CTL_rep2.tif` counts as a control, but so would any name that happens to contain "ctl".

If the folder has no control files, the floors are 0.05 / 0.03.

### Without controls (`USE_CONTROLS = False`, or a single image as input)

The floors are 0, so each image is thresholded only by its own μ+3σ / μ+1.5σ.

This is useful when no controls were imaged. Be aware that an image where most cells have many foci will raise its own threshold, which can lead to undercounting compared with a control-based run.

## Outputs

```
OUTPUT_DIR/
├── summary.csv                      # one row per image: cells, foci_total
└── <image name>/
    ├── <image name>_overlay.png       # cell outlines + detected foci
    ├── <image name>_seeds_overlay.png # seed peaks (x) before watershed
    └── <image name>_per_cell.csv      # foci count per cell
```

## Reporting issues

Open an issue on this repository and include the settings you used and, if possible, an example image.

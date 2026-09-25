# Glare and color-input review

Reviewed on 2026-09-10. The local repository tracks boats and estimates distance.
The separate deployed `/root/HydroRL` pipeline proposes and decodes AprilTags.
Their checkpoints, labels and accuracy figures describe different tasks.

## Observations from the server

Source: `192.168.1.104:/root/shipcaps/20260910-140833/ShipCam0.mp4`.
All 1,470 archived frames decoded; every 60th frame was measured (25 samples).
At the brightest sampled frame, 120, 1.202% of pixels had all BGR channels >=250;
median luminance was 66/255. The inspected scene shows bright overhead light and
reflections in a pool, rather than a uniformly overexposed image. This archived
1280x720 MPEG-4 is not the original full-resolution BGR capture: its pixels cannot
prove sensor clipping.

The recording's `pipeline.json` selects `detector=apriltag`, BGR capture and
CLAHE disabled. The configured proposer is not used in that detector mode.

A separate replay of the server's `yildiz.pt` on the same 25 archived frames,
CPU FP32, `imgsz=640`, confidence 0.05 and IoU 0.7 produced:

| Input | Frames with proposals | Total proposals |
| --- | ---: | ---: |
| Color | 0 | 0 |
| BGR converted to gray, then replicated into three channels | 4 | 4 |

The gray proposals had confidence 0.060–0.181. Inspected proposals on frames 180
and 600 cover reflections/background rather than visible AprilTags. This is not
an accuracy benchmark: complete manual ground truth and ID decoding were not
performed. It argues against changing this color-trained proposer to grayscale
without validation; it does not compare separately trained monochrome models.

Detailed local artifacts are in `results/glare_20260910_140833/`, including
`frames.csv`, `color_gray.csv`, the model hash and saved review images.

## Repeat the brightness analysis

```sh
venv/bin/python analyze_glare.py \
  --video shipcaps_remote/20260910-140833/ShipCam0.mp4 \
  --out results/glare_review_new \
  --sample-every 30 --threshold 250 --top-k 8
```

Use a new output directory. Optional `--roi 0,0.45,1,1` measures the lower frame;
check the actual horizon first. This ROI only controls the analysis, not camera
AE. `--max-frames 300` limits decoding. Output PNGs retain the decoded pixels,
without CLAHE, overlays or resizing. Memory is bounded by `--top-k`.

`near_white_fraction` counts pixels whose three channels reach the threshold;
`exact_white_fraction` counts exact white; `any_channel_high_fraction` counts
pixels with at least one high channel; `largest_near_white_fraction` measures the
largest connected bright area relative to the ROI. All fractions range from
0 to 1. White objects and clouds also trigger these metrics. Review the images
before identifying glare or excluding a training example.

## Exposure and optics

The current cam0 configuration read from the server allows 500–65,487 µs,
analog gain 1–16 and digital gain 1–8, with 0 EV compensation, an empty AE region
and unlocked AE. These are permitted ranges, not measured exposure values for
individual recorded frames. The deployed `camera_settings.py` correctly converts
microseconds to nanoseconds before passing them to Argus.

Argus defines exposure in nanoseconds; an empty AE region lets the device choose
a region, rather than guaranteeing uniform full-frame metering. See
[NVIDIA Settings](https://docs.nvidia.com/jetson/archives/r35.1/ApiReference/Settings_8h_source.html)
and [IAutoControlSettings](https://docs.nvidia.com/jetson/archives/r36.3/ApiReference/classArgus_1_1IAutoControlSettings.html).
An ROI over darker water may increase the exposure chosen by AE; verify this
experimentally. Server comments record worse tag decoding at very short shutter
limits, so applying 2–4 ms blindly is not justified.

A polarizer can suppress reflections, with effectiveness depending on angle
and orientation. Reflection from water is not fully polarized at every viewing
angle. See [Edmund Optics](https://www.edmundoptics.com/knowledge-center/application-notes/optics/introduction-to-polarization).
CLAHE cannot restore clipped information and remains disabled by default.
HDR/WDR support for the deployed sensor/driver combination was not established.

Server code, exposure settings and existing dataset labels were not modified.
See `AUDIT.md` for the wider code/data audit and prepared fixes.

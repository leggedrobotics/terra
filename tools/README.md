# Tools

## Screenshot Grid

Create the 4x2 screenshot grid with circular upper-left number labels:

```bash
conda run -n terra python terra/tools/screenshot_grid.py terra/tools/screenshots terra/tools/screenshots_grid_4x2.png --grid 4x2 --indices 1,2,4,5,6,7,9,10 --no-mark --label --label-template {source} --label-position top-left --label-shape circle --label-circle-scale 1.25 --label-reference-size 600 --label-base-font-size 32 --padding 8 --margin 8
```

The selected indices include the first and last screenshots, with the remaining slots filled from the sorted screenshot sequence.

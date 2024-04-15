# proj-borderpoints
Detection and classification of the border points on the cadastral map of the canton of Fribourg

If running for the 1st time:

```
python .\scripts\data_preparation\pct_to_rgb.py .\config\config_obj_detec.yaml
```

If some overlap between tiles is required:

```
python scripts/data_preparation/get_point_bbox_size
```

The configuration is the one from `prepare_data.py`.<br>
It produces a csv file with the info about the max size of border points at each scale.

General workflow

```
python .\scripts\prepare_data.py .\config\config_obj_detec.yaml
```
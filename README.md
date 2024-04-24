# proj-borderpoints
Detection and classification of the border points on the cadastral map of the canton of Fribourg

If some overlap between tiles is required:

```
python scripts/data_preparation/get_point_bbox_size.py
```

The configuration is the one from `prepare_data.py`.<br>
It produces a csv file with the info about the max size of border points at each scale.

**General workflow**

Data preparation of the areas with GT:

```
python scripts/prepare_data.py config/config_obj_detec.yaml
```

Data preparation for whole tiles without considering any GT:

```
python scripts/prepare_whole_tiles.py config/config_obj_detec.yaml
```

_Warning_: for the script `prepare_whole_tiles.py`, it is necessary to comment and uncomment the proper lines depending if we work with the GT or the other plans.
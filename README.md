# proj-borderpoints
Classification of the border points from the BDMO2 based on the cadastral map of the canton of Fribourg.


## Installation

Use the installation of the STDL's OD.


## General workflow


The workflow can be divided into three parts:

* Data preparation: call of the right script for the preprocessing, _i.e._ `prepare_data.py` to work with ground truth produced over defined bounding boxes and `prepare_whole_tiles.py` to work with entire tiles. More precisely, the following steps are perfomed:
    - Tranform the maps from a color map to RGB images,
    - If ground truth is available, format the labels according to the requirements of the STDL's object detector and clip the maps to the bounding box of the ground truth,
    - Generate a vector layer with the information of the subtiles dividing the maps into square tiles of 512 or 256 pixels,
    - Clip the map to the subtiles.
* Detection of the border points with the STDL's object detector. The necessary documentation is available in the [associated GitHub repository](https://github.com/swiss-territorial-data-lab/object-detector)
* Post-processing: produce one file with all the detections formatted after the experts' requirements.
    - `post_processing.py`: the detections are filtered by their confidance score and ...
    - `point_matching.py`: the detections are matched with the points of the cadastral surveying for areas where it is not fully updated yet,
    - `check_w_land_cover.py`: use the data on land cover to assign the class "non-materialized point" to undetermined points in building and stagnant waters.
    - `heatmap.py`: highlight areas with a high concentration of false positive points.

If some overlap between tiles is required:

```
python scripts/sandbox/get_point_bbox_size.py config/config_sandbox.yaml
```

It produces a csv file with the info about the maximum size of border points at each scale. The file must then be passed as parameter in the data preparation.

**Dataset with GT**

```
python scripts/prepare_data.py config/config_w_gt.yaml
stdl-objdet generate_tilesets config/config_w_gt.yaml
stdl-objdet train_model config/config_w_gt.yaml
stdl-objdet make_detections config/config_w_gt.yaml
stdl-objdet assess_detections config/config_w_gt.yaml
```

The post-processing can be performed and the detections assessed again with the following commands:

```
python scripts/post_processing/post_processing.py config/config_w_gt.yaml
python scripts/assess_by_tile.py config/config_w_gt.yaml
```

In the configuration file, the parameters `keep_datasets` must be set to `False` to preserve the split of the training, validation and test dataset.

Performing the point matching is possible with the ground truth. However, the polygons are then transformed to point and a new script would be needed for the assessement.

```
python scripts/post_processing/point_matching.py config/config_w_gt.yaml
python scripts/post_processing/check_w_land_cover.py config/config_w_gt.yaml
```

**Whole tiles**

```
python scripts/prepare_whole_tiles.py config/config_whole_tiles.yaml
stdl-objdet generate_tilesets config/config_whole_tiles.yaml
stdl-objdet make_detections config/config_whole_tiles.yaml
python scripts/post_processing/post_processing.py config/config_whole_tiles.yaml
python scripts/post_processing/point_matching.py config/config_whole_tiles.yaml
python scripts/post_processing/check_w_land_cover.py config/config_whole_tiles.yaml
python scripts/post_processing/heatmap.py config/config_whole_tiles.yaml
```

The command lines above use the configuration files for the maps with GT areas. The configuration file `config_whole_oth_tiles.yaml` was used for maps on which no point was digitized as part of the ground truth. Only the path to the different folders should change between the two configurations.

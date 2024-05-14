# proj-borderpoints
Detection and classification of the border points on the cadastral map of the canton of Fribourg

If some overlap between tiles is required:

```
python scripts/data_preparation/get_point_bbox_size.py
```

The configuration is the one from `prepare_data.py`.<br>
It produces a csv file with the info about the max size of border points at each scale.

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
    - `point_matching.py`: the detections are matched with the points of the cadastral surveying for area where it is not fully updated yet.


**Dataset with GT**

```
python scripts/prepare_data.py config/config_obj_detec.yaml
stdl-objdet generate_tilsets config/config_obj_detec.yaml
stdl-objdet train_model config/config_obj_detec.yaml
stdl-objdet make_detections config/config_obj_detec.yaml
stdl-objdet assess_detections config/config_obj_detec.yaml
```

**Dataset without GT**

```
python scripts/prepare_whole_tiles.py config/config_whole_tiles.yaml
stdl-objdet generate_tilsets config/config_whole_tiles.yaml
stdl-objdet make_detections config/config_whole_tiles.yaml
python scripts/post_processing.py config/config_whole_tiles.yaml
python scripts/point_matching.py config/config_whole_tiles.yaml
```

The command lines above use the configuration files for the maps with GT areas. The configuration file `config_whole_oth_tiles.yaml` was used for maps on which no points was digitized as par of the ground truth. Only the path to the different folders should change between the two configurations.
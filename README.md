# proj-borderpoints
Classification of the border points from the BDMO2 based on the cadastral map of the canton of Fribourg.


## Installation

The installation is performed from this folder with the following steps:

* Clone the [STDL's object detector](https://github.com/swiss-territorial-data-lab/object-detector),
* Get into the `object-detector` folder,
* Switch to my branch,
* The dockerfile of this project supposes the existence on the machine of an image called `object-detector-stdl-objdet`. 
    * You can control the image existence by listing the available images with `docker images ls`.
    * If it is not available, build it from the folder of the object detector with `docker compose build`.
    * You can control the installation by running `docker compose run --rm stdl-objdet stdl-objdet -h`.
* Go to the folder `proj-borderpoints`,
* Build docker,
* Run docker,
* Go to `proj-borderpoints` directory in docker.


The corresponding command lines are

```
git clone https://github.com/swiss-territorial-data-lab/object-detector.git
cd object-detector
git checkout gs/code_improvement
cd -
docker compose build
docker compose run --rm borderpoints-dev
cd proj-borderpoints            # Command to run in the docker bash
```

**All workflow commands are supposed to be launched in Docker from the proj-borderpoint directory.**

## General workflow


The workflow can be divided into three parts:

* Data preparation: call of the appropriate preprocessing script, _i.e._ `prepare_data.py` to work with ground truth produced over defined bounding boxes and `prepare_whole_tiles.py` to work with entire tiles. More precisely, the following steps are performed:
    - Transform the maps from a color map to RGB images,
    - If ground truth is available, format the labels according to the requirements of the STDL's object detector and clip the maps to the bounding box of the ground truth,
    - Generate a vector layer with the information of the subtiles dividing the maps into square tiles of 512 or 256 pixels,
    - Clip the map to the subtiles.
* Detection of the border points with the STDL's object detector: the necessary documentation is available in the [associated GitHub repository](https://github.com/swiss-territorial-data-lab/object-detector)
* Post-processing: produce one file with all the detections formatted after the experts' requirements.
    - `post_processing.py`: the detections are filtered by their confidence score and ...
    - `point_matching.py`: the detections are matched with the points of the cadastral surveying for areas where it is not fully updated yet,
    - `check_w_land_cover.py`: use the data on land cover to assign the class "non-materialized point" to undetermined points in building and stagnant waters.
    - `heatmap.py`: highlight areas with a high concentration of false positive points.

If some overlap between tiles is required:

```
python scripts/sandbox/get_point_bbox_size.py config/config_sandbox.yaml
```

It produces a csv file with the info about the maximum size of border points at each scale. This maximum size at each scale is then used as the overlap distance for the tile production. The file must then be passed as parameter in the data preparation.

**Dataset with GT**

```
python scripts/instance_segmentation/prepare_data.py config/config_w_gt.yaml
stdl-objdet generate_tilesets config/config_w_gt.yaml
stdl-objdet train_model config/config_w_gt.yaml
stdl-objdet make_detections config/config_w_gt.yaml
stdl-objdet assess_detections config/config_w_gt.yaml
```

The post-processing can be performed and the detections assessed again with the following commands:

```
python scripts/post_processing/post_processing.py config/config_w_gt.yaml
python scripts/instance_segmentation/assess_w_post_process.py config/config_w_gt.yaml
```

In the configuration file, the parameters `keep_datasets` must be set to `True` to preserve the split of the training, validation and test datasets.

Performing the point matching is possible with the ground truth. However, the polygons are then transformed to points and a new script would be needed for the assessment.

```
python scripts/post_processing/point_matching.py config/config_w_gt.yaml
python scripts/post_processing/check_w_land_cover.py config/config_w_gt.yaml
```

**Whole tiles**

```
python scripts/instance_segmentation/prepare_whole_tiles.py config/config_whole_tiles.yaml
stdl-objdet generate_tilesets config/config_whole_tiles.yaml
stdl-objdet make_detections config/config_whole_tiles.yaml
python scripts/post_processing/post_processing.py config/config_whole_tiles.yaml
python scripts/post_processing/point_matching.py config/config_whole_tiles.yaml
python scripts/post_processing/check_w_land_cover.py config/config_whole_tiles.yaml
python scripts/post_processing/heatmap.py config/config_whole_tiles.yaml
```

The command lines above use the configuration files for the maps with GT areas. The configuration file `config_whole_oth_tiles.yaml` was used for maps on which no point was digitized as part of the ground truth. Only the path to the different folders should change between the two configurations.
# Data processing

This README file contains instructions to compute and process RAFT optical flows, Zoe-Depth and Dino-V2 feature matchings for optimizing.

## Data format
The input video data should be organized in the following format:
```
├──sequence_name/
    ├──color/
        ├──00000.jpg
        ├──00001.jpg
        .....
    ├──mask/ (optional; only used for visualization purposes)
        ├──00000.png
        ├──00001.png
        ..... 
```
You can download some of our processed sequences [here TODO]() to skip processing and directly start training.

If you want to train on your own video sequence, we recommend you to start with
shorter sequences (<= 150 frames) and lower resolution (<= 480p) to manage computational cost.


## Preparation
The command below moves files to the correct locations and download pretrained models (this only needs to be run once).
```
cd preprocessing/  

mv exhaustive_raft.py filter_raft.py chain_raft.py RAFT/;
cd RAFT; ./download_models.sh; cd ../

mv extract_dino_features.py dino/
```

## Computing and processing flow

Run the following command to process the input video sequence. Please use absolute path for the sequence directory. We only compute **16** neighboring frames for frames, which is much faster then Omnimotion.
```
conda activate omnimotion
python main_processing.py --data_dir <sequence directory> --chain
```
The processing contains several steps:
- computing all pairwise optical flows using `exhaustive_raft.py`
- computing dino features for each frame using `extract_dino_features.py`
- filtering flows using cycle consistency and appearance consistency check using`filter_raft.py`
- (optional) chaining only cycle consistent flows to create denser correspondences using `chain_raft.py`. 
  We found this to be helpful for handling sequences with rapid motion and large displacements. 
  For simple motion, this may be skipped by omitting `--chain` to save processing time. 

After processing the folder should look like the following:
```
├──sequence_name/
    ├──color/
    ├──mask/ (optional; only used for visualization purposes)
    ├──count_maps/
    ├──features/
    ├──raft_exhaustive/
    ├──raft_masks/
    ├──flow_stats.json
```

## Computing Depth 

Run the following command after you got the flow to compute the depth and recompute the masks.
```
python create_depth.py --data_dir <sequence directory>
```


## Compute Long-term Feature Matching
Run the following command whenever you got the sequence images.
First inference the Dino-v2 feature maps for ecah image. Please specify the image size in the `get_featmap()` function.
```
python inf_dinov2.py --data_dir <sequence directory>
```
Then compute and save the matched sparse points. You may change the threshold of the `matchpair()` function parameters to have different matching result.
```
python featmatch.py --data_dir <sequence directory>
```
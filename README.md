# mocast_nus
Private repo containing development code for a motion prediction model for autonomous vehicles trained and evaluated on NuScenes dataset

## Setup
```git clone git@github.com:rodgrac/mocast_nus.git```

```cd mocast_nus```

Required packages in ```requirements.txt```

### NuScenes devkit:

```mkdir -p datasets/nuScenes && cd $_```

```git clone https://github.com/nutonomy/nuscenes-devkit.git```

### Preprocessed dataset paths

Mini train and val:

```/scratch/rodney/datasets/nuScenes/processed/nuscenes-v1.0-mini-train.h5```
```/scratch/rodney/datasets/nuScenes/processed/nuscenes-v1.0-mini-val.h5```

Full train and val:

```/scratch/rodney/datasets/nuScenes/processed/nuscenes-v1.0-trainval-train.h5```
```/scratch/rodney/datasets/nuScenes/processed/nuscenes-v1.0-trainval-val.h5```

Trained models saved in ./models directory

### Meta-learning

Branch used: metalr

```cd src/meta_lr```

Train: ```python3 meta_train.py```

Test: ```python3 meta_eval.py```





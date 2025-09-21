# Evaluating UniPixel

## 🛠️ Environment Setup

Please refer to [TRAIN.md](/docs/TRAIN.md) for setting up the environment.

## 📚 Checkpoint Preparation

Download the checkpoints from [Hugging Face](https://huggingface.co/collections/PolyU-ChenLab/unipixel-68cf7137013455e5b15962e8) and place them into the `model_zoo` folder.

```
UniPixel
└─ model_zoo
   ├─ UniPixel-3B
   └─ UniPixel-7B
```

## 📦 Dataset Preparation

Download the desired datasets / benchmarks from [Hugging Face](https://huggingface.co/datasets/PolyU-ChenLab/UniPixel-SFT-1M), extract them, and place them into the `data` folder. The processed files should be organized in the following structure (taking `ref_youtube_vos` as an example).

```
UniPixel
└─ data
   └─ ref_youtube_vos
      ├─ meta_expressions
      ├─ train
      ├─ valid
      └─ mask_dict.pkl
```

## 🔮 Start Evaluation

Use the following command to evaluate UniPixel automatically on all benchmarks. The default setting is to distribute the samples to multiple GPUs/NPUs for acceleration.

```shell
bash scripts/auto_eval.sh <path-to-checkpoint>
```

You may comment out some datasets in [`auto_eval.sh`](https://github.com/PolyU-ChenLab/UniPixel/blob/main/scripts/auto_eval.sh) if you don't need them.

The inference outputs and evaluation metrics will be saved into the `<path-to-checkpoint>` folder by default.

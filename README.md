# Super-resolution of biomedical volumes with 2D supervision

CVPR Workshop on Computer Vision for Microscopy Image Analysis (CVMI) 2024

[**Website**](https://mlins.org/msdsr/) /
[**arXiv**](https://arxiv.org/abs/2404.09425) /
[**MLiNS Lab**](https://mlins.org/) /
[**OpenSRH**](https://github.com/MLNeurosurg/opensrh) /
[**Model checkpoint**](https://www.dropbox.com/scl/fi/3c5nd7gax3a67o78kdatz/msdsr_cvmi24_checkpoint.tgz?rlkey=t455yr8led8e81dir5cmchiy2&dl=0) /
[**Sample volumes**](https://www.dropbox.com/scl/fo/vb4qrs9jm8duymbjp7t4s/AF0eIYF_Jd0hg0pMzR-qPWc?rlkey=kezu8shqsx6wgcd0lh9xtl1h9&dl=0)

## Installation

1. Clone MSDSR github repo
    ```console
    git clone git@github.com:MLNeurosurg/msdsr.git
    ```
2. Install miniconda: follow instructions
    [here](https://docs.conda.io/en/latest/miniconda.html)

3. Create conda environment:
    ```console
    conda create -n msdsr python=3.11
    ```
4. Activate conda environment:
    ```console
    conda activate msdsr
    ```
6. Install package and dependencies
    ```console
    pip install -e .
    ```

## Model checkpoint and sample volumes
We release our pretrained DDPM model checkpoint and sample 3D volumes. They are available at the links below:

[**Model checkpoint**](https://www.dropbox.com/scl/fi/3c5nd7gax3a67o78kdatz/msdsr_cvmi24_checkpoint.tgz?rlkey=t455yr8led8e81dir5cmchiy2&dl=0) /
[**Sample volumes**](https://www.dropbox.com/scl/fo/vb4qrs9jm8duymbjp7t4s/AF0eIYF_Jd0hg0pMzR-qPWc?rlkey=kezu8shqsx6wgcd0lh9xtl1h9&dl=0)

## Training / evaluation instructions

The code base is written using PyTorch Lightning, with custom network and datasets for OpenSRH.

To train MSDSR on the OpenSRH dataset:

1. Download OpenSRH - request data [here](https://opensrh.mlins.org).
2. Update the sample config file in `train/config/train_msdsr.yaml` with
    desired configurations.
3. Change directory to `train` and activate the conda virtual environment.
4. Use `train/train_msdsr.py` to start training:
    ```console
    python train_msdsr.py -c=config/train_msdsr.yaml
    ```

To evaluate with your trained model:
1. Update the sample config files in `eval/config/*.yaml` with
    the checkpoint path and other desired configurations per file.
    If you are using the released checkpoint, place the checkpoint in the path `$log_dir/$exp_name/msdsr_cvmi24/models/d17986ac.ckpt`.
2. Change directory to `eval` and activate the conda virtual environment.
3. Use the evaluation scripts in `eval/*.py` for evaluation. For example:
    ```console
    # paired 2D evaluation
    python eval_paired.py -c=config/eval_paired.yaml

    # generate 3D volumes evaluation
    python generate_volumes.py -c=config/generate_volumes.yaml

    # generate metrics for 3D volumes [require paired 2D evaluation and 3D volume generated]
    python compute_volume_metrics.py -c=config/compute_metrics.yaml
    ```

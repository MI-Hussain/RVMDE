# RVMDE

Official implementation of "RVMDE : Radar Validated Monocular Depth Estimation for Robotics", https://arxiv.org/abs/2109.05265v1.


## :wrench: Dependencies and Installation

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.9](https://pytorch.org/)

### Installation

1. Clone repo

    ```bash
    git clone https://github.com/MI-Hussain/RVMDE
    ```

1. Install dependent packages

```bash
pypardiso
tensorboardX
nuscenes-devkit
```
## Dataset

Download [nuScenes dataset](https://www.nuscenes.org/) (Full dataset (v1.0)) into data/nuscenes/

## Directories
```plain
rvmde/
    data/                           							 
        nuscenes/                 		    
                annotations/
                maps/
                samples/
                sweeps/
                v1.0-trainval/
    dataloader/
    list/
    result/
    model/                   				   	        
                   	     				
```

### Dataset Prepration use the externel repos

Please follow external repos (https://github.com/lochenchou/DORN_radar) for Height Extension and (https://github.com/longyunf/rc-pda) for RVMDE with MER's to generte the dataset for training and evaluation.

### Evaluation for RVMDE on nuScenes

Download [pre-trained weights](https://drive.google.com/file/d/1VKVg63d5UMNjc2busvdM23rXrs8TZb-X/view?usp=sharing)


Modifying dataset path in `valid_loader.py`, evalutation list path in `data_loader.py`, pretrained_weights path in Evalutation_rvmde.py file to evalute. 

For evaluation of day,night,rain change the list path first. The evaluation lists are saved in .\list directory.

``` bash
Evaluate_rvmde.ipynb                  #Evaluation
```

#### Evaluation for RVMDE with MERs on nuScenes

Please visit this work (https://github.com/longyunf/rc-pda) for detail information of data prepration of training and evaluation sets. 

Download [pre-trained RVMDE with MERs weights](https://drive.google.com/file/d/1AOFOnXd6USSb8-kYzGAtxI9gUdPfl15m/view?usp=sharing)


``` bash
Evaluation_RVMDE_with_MERS.ipynb      #Evaluation
```

## Citation
```plain
@Article{hussain2021rvmde,
    title={RVMDE : Radar Validated Monocular Depth Estimation for Robotics},
    author={Muhammad Ishfaq Hussain, Muhammad Aasim Rafique and Moongu Jeon},
    journal={arXiv:2109.05265v1},
    year={2021}
}
```

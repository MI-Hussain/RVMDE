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
Evaluation_rvmde.ipynb                  #Evaluation
```

### Evaluation for RVMDE with MERs on nuScenes

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

The following works have been used by RVMDE:

  ~~~
@InProceedings{Long_2021_CVPR,
    author    = {Long, Yunfei and Morris, Daniel and Liu, Xiaoming and Castro, Marcos and Chakravarty, Punarjay and Narayanan, Praveen},
    title     = {Radar-Camera Pixel Depth Association for Depth Completion},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {12507-12516}
}

@INPROCEEDINGS{9506550,
  author={Lo, Chen-Chou and Vandewalle, Patrick},
  booktitle={2021 IEEE International Conference on Image Processing (ICIP)}, 
  title={Depth Estimation From Monocular Images And Sparse Radar Using Deep Ordinal Regression Network}, 
  year={2021},
  pages={3343-3347},
  doi={10.1109/ICIP42928.2021.9506550}
}
```

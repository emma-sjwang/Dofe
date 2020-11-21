# DoFE: Domain-oriented Feature Embeddingfor Generalizable Fundus Image Segmentationon Unseen Datasets
by [Shujun Wang](https://www.cse.cuhk.edu.hk/~sjwang), [Lequan Yu](https://yulequan.github.io/), Kang Li, [Yang Xin](https://xy0806.github.io/), [Chi-Wing Fu](http://www.cse.cuhk.edu.hk/~cwfu/), and [Pheng-Ann Heng](http://www.cse.cuhk.edu.hk/~pheng/).

## Introduction
This repository is for our TMI2020 paper '[DoFE: Domain-oriented Feature Embeddingfor Generalizable Fundus Image Segmentationon Unseen Datasets](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9163289)'.
![cellgraph](https://emmaw8.github.io/project_img/TMI2020-dofe.png)
Schematic  diagram  of  our  proposedDoFEframework  to  utilize  multi-source  domain  datasets {D1,...,DK} forgeneralizable segmentation on the unseen dataset D_{K+1}. We adopt the Domain Knowledge PoolMpoolto learn and memorizethe  multi-source  domain  prior  knowledge.  Importantly,  our  framework  dynamically  enriches  the  image  semantic  featurehswith domain-oriented aggregated featureˆhaggextracted fromMpoolto improve the expressiveness of the semantic feature.

## Requirements
-   python 3.6.8
   
   ``` bash
   conda create -n DOFE python=3.6.8 
   ```
   
-   PyTorch 1.5.0 
    
    ``` bash
    conda activate DOFE 
    conda install pytorch==1.5.0 torchvision cudatoolkit=9.2 -c pytorch 
    pip install tensorboardX==2.0
    pip install opencv-python
    pip install pyyaml
    pip install MedPy
    conda install -c anaconda scikit-image
    ```
    

## Usage
1. Clone the repository and download the [dataset](https://drive.google.com/file/d/1p33nsWQaiZMAgsruDoJLyatoq5XAH-TH/view?usp=sharing) into your own folder and change `--data-dir` correspondingly.

2. Train the model.

    ``` bash
    python train.py -g 0 --datasetTrain 1 2 3 --datasetTest 4 --batch-size 16 --resume ./pretrained-weight/test4-epoch40.pth.tar # You need to pretrain a model
    ```
3. Test the model.

    ``` bash
    python test.py --model-file ./logs/test4/lam0.9/20201120_215812.079473/checkpoint_80.pth.tar --datasetTest 4 -g 0

    ```

## Citation
If DoFE is useful for your research, please consider citing:
```angular2html
@article{wang2020dofe,
  title={DoFE: Domain-oriented Feature Embedding for Generalizable Fundus Image Segmentation on Unseen Datasets},
  author={Wang, Shujun and Yu, Lequan and Li, Kang and Yang, Xin and Fu, Chi-Wing and Heng, Pheng-Ann},
  journal={IEEE Transactions on Medical Imaging},
  year={2020},
  publisher={IEEE}
}
```



# MA-TransUNet
This repo holds code for “MA-TransUnet: U-shaped Transformer with Multi-Scale CNN-based Auxiliary Network for Medical Image Segmentation” []()

## Usage

### 1. Prepare data
Access to the synapse multi-organ dataset:
* Download the synapse dataset from the [official website](https://www.synapse.org/#!Synapse:syn3193805/wiki/). Convert them to numpy format, clip the images within [-125, 275], normalize each 3D image to [0, 1], and extract 2D slices from 3D volume for training cases while keep the 3D volume in h5 format for testing cases.
* Or you can just use the [processed data](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd) ,the Synapse datasets we used are provided by TransUnet's authors.Please go to [link](https://github.com/Beckschen/TransUNet) for details, or please send an Email to jienengchen01 AT gmail.com to request the preprocessed data. If you would like to use the preprocessed data, please use it for research purposes and do not redistribute it (following the TransUnet's License). Please prepare data in the data directory:
```
├── MA-TransUNet
    ├── data
    │    └──Synapse
    │          ├── test_vol_h5
    │          │     ├── case0001.npy.h5
    │          │     └── *.npy.h5
    │          └── train_npz
    │                ├── case0005_slice000.npz
    │                └── *.npz
    └── lists
         └──lists_Synapse
               ├── all.lst
               ├── test_vol.txt
               └── train.txt
        
```
### 2. Environment

Please prepare an environment with python>=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

### 3. Train/Test

- Train

```bash
MA_TransUnet_S:
python train.py --root_path ./data/Synapse/train_npz --test_path ./data/Synapse/test_vol_h5 --batch_size 20 --eval_interval 20 --max_epochs 450 --img_size 224 --module networks.MA_TransUnet_S.myFormer --output_dir './model_out_S' 
  ...
  ...
  ...
MA_TransUnet_E:
python train.py --root_path ./data/Synapse/train_npz --test_path ./data/Synapse/test_vol_h5 --batch_size 20 --eval_interval 20 --max_epochs 450 --img_size 224 --module networks.MA_TransUnet_E.myFormer --output_dir './model_out_E'
```

- Test 
* You can download the learning weights of our network from the [link](https://drive.google.com/file/d/1gGxThlsHwzm7a_8Ocvlk6ptoU9AlWvbG/view?usp=sharing).
```bash
python test.py --volume_path ./data/Synapse/ --output_dir './model_out/model_out_E/ef' --max_epochs 450 --img_size 224 --is_savenii
```


## Reference

## Citations

```bibtex

```

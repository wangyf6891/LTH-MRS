# LTH-MRS
This is an implemention for our IJIS(International Journal of intelligent Systems) special issue paper "Exploring Lottery Ticket Hypothesis in Media Recommender Systems" based on pytorch. [Paper in arXiv.](https://arxiv.org/abs/2108.00944)


## Requirements
+ pytorch == 1.4.0+cu100
+ dgl == dgl-cu100
+ numpy == 1.19.2
+ python3

## Datasets

+ Yelp2018: We provide the processed dataset in [data file](/data).
+ Kwai and TikTok: we provide the URL of original data. We do not provide the processed datasets here because we don't make sure whether we have the right to release them. If you have difficulties getting them or processing them, you can contact us.
  [Kwai original data](https://www.kuaishou.com/activity/uimc); [TikTok original data](https://www.biendata.xyz/competition/icmechallenge2019/).

## Parameters
Key parameters are mentioned in our paper. Please refer to the paper. Other parameters are all listed in the [parse_args part](body.py).


## Commands 
We provide the following commands for our models and baselines.

### LightGCN-DGL
We implemented the LightGCN model using DGL, which you can run directly in code:

  ```
  python train.py
  ```

DGL is used to avoid large matrix operation and accelerate model training and reasoning. You can also refer to the original implementation of LightGCN: [LightGCN-Pytorch-Matrix Operation](https://github.com/kuandeng/LightGCN)

#### 1. IMP:
There are two backbone models in our paper, MF and LightGCN. You can run IMP- LightGCN as the following:

  ```
  python train_emb.py --dataset yelp2018 --embedding_size 32 --reg_weight 1e-4 --pruning_percent_wei 0.1 --epoch 1000
  ```

And you can run IMP-MF with this command line:

  ```
  python train_emb_mf.py --dataset yelp2018 --embedding_size 32 --reg_weight 1e-4 --pruning_percent_wei 0.1 --epoch 1000
  ```
 
#### 2. OMP:
As mentioned before, you can run OMP algorithm as the following:

  ```
python train_emb_oneshot.py --dataset yelp2018 --embedding_size 32 --reg_weight 1e-4 --pruning_percent_wei 0.1 --epoch 1000
  ```

  ```
python train_emb_mf_oneshot.py --dataset yelp2018 --embedding_size 32 --reg_weight 1e-4 --pruning_percent_wei 0.1 --epoch 1000
  ```

#### 3. RP:
You can run RP algorithm as the following:

  ```
python train_emb_random.py --dataset yelp2018 --embedding_size 32 --reg_weight 1e-4 --pruning_percent_wei 0.1 --epoch 1000
  ```

  ```
python train_emb_mf_random.py --dataset yelp2018 --embedding_size 32 --reg_weight 1e-4 --pruning_percent_wei 0.1 --epoch 1000
  ```


#### 4. Without_rewinding:
You can run the without rewinding algorithm as the following:

  ```
python train_emb_wit.py --dataset yelp2018 --embedding_size 32 --reg_weight 1e-4 --pruning_percent_wei 0.1 --epoch 1000
  ```

  ```
python train_emb_mf_wit.py --dataset yelp2018 --embedding_size 32 --reg_weight 1e-4 --pruning_percent_wei 0.1 --epoch 1000
  ```

#### 5. PEP:
you can run the command line: 

  ```
python train_pep.py
  ```

And choose the backbone models inside the code of this part.

#### 6. LCM:
You can refer to the code [LCM](https://github.com/gusye1234/KD_on_Ranking) for this part. The data input format and evaluation indicators in this code are different from our code. If you have any difficult to modify them, please contact us. Here we provide the [Yelp2018_LCM](/data/Yelp2018_LCM) suitable for this code input format

## Notation

There are four pruning rules for IMP-MRS in our code. This paper only studies "the method of pruning the representation of user and item together ". That means we use  ```
--part == join
	```
in our method.

## Citation
If you use our codes in your research, please cite our paper.

  ```
@article{wang2021exploring,
  title={Exploring Lottery Ticket Hypothesis in Media Recommender Systems},
  author={Wang, Yanfang and Sui, Yongduo and Wang, Xiang and Liu, Zhenguang and He, Xiangnan},
  journal={International Journal of intelligent Systems},
  year={2021}
  doi={10.1002/int.22827}
}
  ```



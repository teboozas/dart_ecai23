# Towards Flexible Time-to-event Modeling: Optimizing Neural Networks via Rank Regression (ECAI 2023)

This repository contains the official implementation of the Deep AFT Rank-regression model for Time-to-event prediction (DART) of the paper **[Towards Flexible Time-to-event Modeling: Optimizing Neural Networks via Rank Regression](https://arxiv.org/abs/2307.08044)** by Authors: Hyunjun Lee*, Junhyun Lee*, Taehwa Choi, Jaewoo Kang and Sangbum Choi.
(* Equal contribution)

![image](./DART.png)

## Abstract
Time-to-event analysis, also known as survival analysis, aims to predict the time of occurrence of an event, given a set of features. 
One of the major challenges in this area is dealing with censored data, which can make learning algorithms more complex. 
Traditional methods such as Cox's proportional hazards model and the accelerated failure time (AFT) model have been popular in this field, but they often require assumptions such as proportional hazards and linearity.
In particular, the AFT models often require pre-specified parametric distributional assumptions.
To improve predictive performance and alleviate strict assumptions, there have been many deep learning approaches for hazard-based models in recent years.  
However, representation learning for AFT has not been widely explored in the neural network literature, despite its simplicity and interpretability in comparison to hazard-focused methods.
In this work, we introduce the Deep AFT Rank-regression model for Time-to-event prediction (*DART*). This model uses an objective function based on Gehan's rank statistic, which is efficient and reliable for representation learning. 
On top of eliminating the requirement to establish a baseline event time distribution, *DART* retains the advantages of directly predicting event time in standard AFT models.
The proposed method is a semiparametric approach to AFT modeling that does not impose any distributional assumptions on the survival time distribution. 
This also eliminates the need for additional hyperparameters or complex model architectures, unlike existing neural network-based AFT models. 
Through quantitative analysis on various benchmark datasets, we have shown that *DART* has significant potential for modeling high-throughput censored time-to-event data.

## Description
The code in this repository is used to perform survival analysis using a deep learning approach. It uses a variety of libraries such as PyTorch, lifelines, pandas, and numpy to preprocess the data, build the model, and evaluate its performance.


## Installation
To run the code, you will need to install the following Python libraries:

- numpy
- pandas
- pycox
- scikit-learn
- sklearn-pandas
- torch
- torchtuples
- lifelines
- wandb (optional)

You can install these libraries using pip:

```bash
pip3 install numpy pandas scikit-learn sklearn-pandas torch torchtuples lifelines pycox wandb
```


## Usage

Here is an example of how to run the script:

```bash
python DART_kkbox.py --dataset=kkbox_v1 --lr=1e-5 --num_layers=6 --num_nodes=256 --batch_size=1024 --use_BN --wandb
```


## Data

The script expects to find the data in a pickle file in the `./data/` directory. In case of KKBox, the data should be in the form of a pandas DataFrame with the following columns:

- `n_prev_churns`
- `log_days_between_subs`
- `log_days_since_reg_init`
- `age_at_start`
- `log_payment_plan_days`
- `log_plan_list_price`
- `log_actual_amount_paid`
- `is_auto_renew`
- `is_cancel`
- `strange_age`
- `nan_days_since_reg_init`
- `no_prev_churns`
- `city`
- `gender`
- `registered_via`

## Citation
TBA

## Contributing
Contributions are welcome. Please open an issue to discuss your ideas or submit a pull request with your changes.

## License
This project is licensed under the terms of the MIT license.








# Forecasting and time series in PyTorch (pyFAST)

## (1) Install

## (2) Examples

## (3) License

## (4) Coding style

    The linux coding style.
    
    (a) Add variable types on (member) functions.
    
    (b) How to do documentation?
    
## (5) Potential research topics.

    (a) Generative learning for infectious disease case estimation from coarse-granulary to fine-granulary

        > Given fine-granulary data, can we estimate coarse-granulary data?

          Some situations may require this, e.g., when we have fine-granulary data but we want to estimate coarse-granulary data.

        > Given coarse-granulary data, can we estimate fine-granulary data?
    
    (b) instance normalization vs. generative pretraining

        > pretraining on 'uts_dataset + instance scale' can get good inferences, but worse on generations.

        > generative pretraining on 'uts_dataset + global scale' can get relative good inferences, 
            as well as good generations.

        > for increasing uts_dataset, 'glbal scale' can not be determined directly, 
            but 'instance scale' can be determined directly.

        > what is normalization, and why normalization?

## (6) Notes on meetings

    (a) 2025-01-13
    
        The pyFAST can do time series recovery (imputation), forecasting, and generation.

    (b) data structure

        (b1) mts: x -> [batch_size, window_size, n_vars]
    
        (b2) uts dataset: x -> [batch_size * n_vars, window_size, 1]
    
            learning rate: 需要同时适应多个数据集
    
        (b3) uts dataset: 适应不同的time series length数据，比如电池ts: list of (cycle_id, SoH)
    
        (b4) pretrained modeling / 

    (c) data fusion

       (c1) uts dataset + mask to support data missing situations

       (c2) uts dataset + exogenous time series to support generative pretrained modeling

       (c3) uts dataset + ts_mask + exogenous time series with ex_mask

    (d) mts dataset pyFAST-v3

       (d1) deep reinforcement learning

    2025-01-14

    从llm中embedding | time series variable embedding的问题
    
    word (id) -> embedding | vector (numeric) -> embedding 空间映射
    
    embedding是可以表示word：要去知道与局限

    要好过用embedding来表示多个变量的向量

    这是两件事
    
    我做的实验：用xmcdc的cases数据，三种疾病，同样实验设置下，uts的实验性能更好
    
        一个实验：mts三个变量
        另外一个实验：uts，三个变量作为一个

    2025-01-15 idea reconsideration

    (a)（肝病）全球每年有3000多万新增肝病患者，正常肝病早期筛查方法，到2050做不完全员筛查。我们的方法是通过抽血几可以生成早期预警，预计2033年就可以筛查全球。

    (b) 全国每年新增电池多少Ah, 退役多少Ah，每一Ah回收成本是多少，核心的回收问题在电池状态的检测，通过少量预测即可进行准确电池状态估算，
        以及到2030年，可减排xxx二氧化碳，减少多少xxx加仑的汽油，以及节省多少xxx时间。

        - 全国电池每年的新增量
    
        - 各行各业的使用情况
    
        - 电池回收利用的情况
    
        - Vision


## 2025-01-17 todo
    
    (1). glucose-v1: short-term long-sequnce glucose forecasting, deep learning + cross-dataset training
    
        [sh_diabetes, kdd2018_glucose]
    
        evaluator=Evaluator(['MAE', 'MAPE', 'RMSE', 'PCC'])
    
        -- 森镇
    
            cut to chase: figure / plot -> model training mechnism
    
        -- 陈愉，73-torch2_env + v100
    
    
    (2). glucose-v2: generative learning for glucose estimation
    
        [sh_diabetes, mimic-iii]
    
        evaluator=Evaluator(['MAE', 'MAPE', 'RMSE', 'PCC'])
    
        cut to chase: foods vs. glucose values
    
    
    (3). battery-v1: generative learning for battery status of health estimation
    
        []
    
        evaluator=Evaluator(['MAE', 'MAPE', 'SDRE', 'PCC'])
    
        - generative pretrained modeling + downstream task
    
        - 胡樾：统计数据，应用背景，找几个报告，填写电池各类报告的数据。
    
    
    (4). energy-v11 (pinn_wpf):  pinn / gl wind power estimation
    
        - PDE / ODE
    
        - Physical-information
    
        - la-haute-borne, reduced.csv: improved 500%
    
        - kdd2022-SDWPF
    
        - 胡樾
    

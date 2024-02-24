# Time-Series-Papers
This is a repository for collecting papers and code in time series domain.

## Table of Content  

- [Linear](#linear)
- [RNN and CNN](#rnn-and-cnn)
- [Transformer](#transformer)
- [GNN](#gnn)
- [Framework](#framework)
- [Repositories](#repositories)

```bash
  ├─ Linear/  
  ├─ RNN and CNN/           
  ├─ Transformer/
  ├─ GNN/
  ├─ Framework/                
  └─ Repositories/         
```

---

## Linear

- DLinear: **Are Transformers Effective for Time Series Forecasting**, _Zeng et al._, AAAI 2023. \[[paper](https://arxiv.org/abs/2205.13504)\]\[[code](https://github.com/cure-lab/LTSF-Linear)\]
- **TSMixer: Lightweight MLP-Mixer Model for Multivariate Time Series Forecasting**, _Ekambaram et al._, KDD 2023. \[[paper](https://arxiv.org/abs/2306.09364)\]\[[model](https://huggingface.co/docs/transformers/main/en/model_doc/patchtsmixer)\]\[[example](https://github.com/ibm/tsfm#notebooks-links)\]
- **Tiny Time Mixers (TTMs): Fast Pretrained Models for Enhanced Zero/Few-Shot Forecasting of Multivariate Time Series**, _Ekambaram et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.03955)\]
- **FCDNet: Frequency-Guided Complementary Dependency Modeling for Multivariate Time-Series Forecasting**, _Chen et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.16450)\]\[[code](https://github.com/onceCWJ/FCDNet)\]
- **TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting**, _Wang et al._, ICLR 2024. \[[paper](https://openreview.net/forum?id=7oLshfEIC2)\]\[[code]\]

---

## RNN and CNN

---

## Transformer

- **Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting**, _Zhou et al._, AAAI 2021 Best paper. \[[paper](https://arxiv.org/abs/2012.07436)\]\[[code](https://github.com/zhouhaoyi/Informer2020)\]
- **Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting**, _Wu et al._, NeurIPS 2021. \[[paper](https://arxiv.org/abs/2106.13008)\]\[[code](https://github.com/thuml/Autoformer)\]\[[slides](https://wuhaixu2016.github.io/pdf/NeurIPS2021_Autoformer.pdf)\]
- **Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy**, _Xu et al._, ICLR 2022. \[[paper](https://arxiv.org/abs/2110.02642)\]\[[code](https://github.com/thuml/Anomaly-Transformer)\]\[[slides](https://wuhaixu2016.github.io/pdf/ICLR2022_Anomaly.pdf)\]
- **Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting**, _Liu et al._, NeurIPS 2022. \[[paper](https://arxiv.org/abs/2205.14415)\]\[[code](https://github.com/thuml/Nonstationary_Transformers)\]
- **TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis**, _Wu et al._, ICLR 2023. \[[paper](https://arxiv.org/abs/2210.02186)\]\[[code](https://github.com/thuml/TimesNet)\]\[[slides](https://wuhaixu2016.github.io/pdf/ICLR2023_TimesNet.pdf)\
- **iTransformer: Inverted Transformers Are Effective for Time Series Forecasting**, _Liu et al._, ICLR 2024 Spotlight. \[[paper](https://arxiv.org/abs/2310.06625)\]\[[code](https://github.com/thuml/iTransformer)\]
- **FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting**, _Zhou et al._, ICML 2022. \[[paper](https://arxiv.org/abs/2201.12740)\]\[[code](https://github.com/MAZiqing/FEDformer)\]\[[DAMO-DI-ML](https://github.com/DAMO-DI-ML)\]
- PatchTST: **A Time Series is Worth 64 Words: Long-term Forecasting with Transformers**, _Nie et al._, ICLR 2023. \[[paper](https://arxiv.org/abs/2211.14730)\]\[[code](https://github.com/yuqinie98/PatchTST)\]
- **Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting**, _Zhang and Yan_, ICLR 2023.  \[[paper](https://openreview.net/forum?id=vSVLM2j9eie)\]\[[code](https://github.com/Thinklab-SJTU/Crossformer)\]

---

## GNN

- **A Survey on Graph Neural Networks for Time Series: Forecasting, Classification, Imputation, and Anomaly Detection**, _Jin et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2307.03759)\]\[[code](https://github.com/KimMeen/Awesome-GNN4TS)\]

---

## Framework

- **SimMTM: A Simple Pre-Training Framework for Masked Time-Series Modeling**, _Dong et al._, NeurIPS 2023 Spotlight. \[[paper](https://arxiv.org/abs/2302.00861)\]\[[code](https://github.com/thuml/SimMTM)\]
- **TSPP: A Unified Benchmarking Tool for Time-series Forecasting**, _Bączek et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.17100)\]\[[code](https://github.com/NVIDIA/DeepLearningExamples/tree/master/Tools/PyTorch/TimeSeriesPredictionPlatform)\]
- **One Fits All:Power General Time Series Analysis by Pretrained LM**, _Zhou et al._, NeurIPS 2023. \[[paper](https://arxiv.org/abs/2302.11939)\]\[[code](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All)\]\[[AI-for-Time-Series-Papers-Tutorials-Surveys](https://github.com/DAMO-DI-ML/AI-for-Time-Series-Papers-Tutorials-Surveys)\]
- **Large Language Models Are Zero-Shot Time Series Forecasters**, _Gruver et al._, NeurIPS 2023. \[[paper](https://arxiv.org/abs/2310.07820)\]\[[code](https://github.com/ngruver/llmtime)\]
- **GPT-ST: Generative Pre-Training of Spatio-Temporal Graph Neural Networks**, _Li et al._, NeurIPS 2023. \[[paper](https://arxiv.org/abs/2311.04245)\]\[[code](https://github.com/HKUDS/GPT-ST)\]
- **Lag-Llama: Towards Foundation Models for Time Series Forecasting**, _Rasul et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.08278)\]\[[code](https://github.com/kashif/pytorch-transformer-ts)\]
- TimesFM: **A decoder-only foundation model for time-series forecasting**, _Das et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.10688)\]
- **TimeGPT-1**, _Garza et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.03589)\]\[[nixtla](https://github.com/Nixtla/nixtla)\]
- **Time-LLM: Time Series Forecasting by Reprogramming Large Language Models**, _Jin et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2310.01728)\]
- **AutoTimes: Autoregressive Time Series Forecasters via Large Language Models**, _Liu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.02370)\]
- **Large Models for Time Series and Spatio-Temporal Data: A Survey and Outlook**, _Jin et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.10196)\]\[[code](https://github.com/qingsongedu/Awesome-TimeSeries-SpatioTemporal-LM-LLM)\]
- **MOMENT: A Family of Open Time-series Foundation Models**, _Goswami et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.03885)\]\[[code](https://anonymous.4open.science/r/BETT-773F/)\]
- **Large Language Models for Time Series: A Survey**, _Zhang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.01801)\]\[[code](https://github.com/xiyuanzh/awesome-llm-time-series)\]
- **Unified Training of Universal Time Series Forecasting Transformers**, _Woo et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.02592)\]
- **Multi-Patch Prediction: Adapting LLMs for Time Series Representation Learning**, _Bian et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.04852)\]


---

## Repositories

- \[[Time-Series-Library](https://github.com/thuml/Time-Series-Library)\]
- \[[time-series-transformers-review](https://github.com/qingsongedu/time-series-transformers-review)\]\[[awesome-AI-for-time-series-papers](https://github.com/qingsongedu/awesome-AI-for-time-series-papers)\]\[[Awesome-TimeSeries-SpatioTemporal-LM-LLM](https://github.com/qingsongedu/Awesome-TimeSeries-SpatioTemporal-LM-LLM)\]
- \[[statsforecast](https://github.com/Nixtla/statsforecast)\]


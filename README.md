# Time-Series-Papers
This is a repository for collecting papers and code in time series domain.

## Table of Content  

- [Linear](#linear)
- [RNN and CNN](#rnn-and-cnn)
- [Transformer](#transformer)
- [GNN](#gnn)
- [LLM Framework](#llm-framework)
- [Diffusion Model](#diffusion-model)
- [Benchmark and Dataset](#benchmark-and-dataset)
- [Repositories](#repositories)

```bash
  ├─ Linear/  
  ├─ RNN and CNN/           
  ├─ Transformer/
  ├─ GNN/
  ├─ LLM Framework/
  ├─ Diffusion Model/
  ├─ Benchmark and Dataset/                      
  └─ Repositories/         
```

---

## Linear

- DLinear: **Are Transformers Effective for Time Series Forecasting**, _Zeng et al._, AAAI 2023. \[[paper](https://arxiv.org/abs/2205.13504)\]\[[code](https://github.com/cure-lab/LTSF-Linear)\]
- **TSMixer: Lightweight MLP-Mixer Model for Multivariate Time Series Forecasting**, _Ekambaram et al._, KDD 2023. \[[paper](https://arxiv.org/abs/2306.09364)\]\[[model](https://huggingface.co/docs/transformers/main/en/model_doc/patchtsmixer)\]\[[example](https://github.com/ibm/tsfm#notebooks-links)\]
- **Tiny Time Mixers (TTMs): Fast Pretrained Models for Enhanced Zero/Few-Shot Forecasting of Multivariate Time Series**, _Ekambaram et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2401.03955)\]
- **FCDNet: Frequency-Guided Complementary Dependency Modeling for Multivariate Time-Series Forecasting**, _Chen et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.16450)\]\[[code](https://github.com/onceCWJ/FCDNet)\]
- **TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting**, _Wang et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2405.14616)\]\[[code](https://github.com/kwuking/TimeMixer)\]

---

## RNN and CNN

- **TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis**, _Wu et al._, ICLR 2023. \[[paper](https://arxiv.org/abs/2210.02186)\]\[[code](https://github.com/thuml/TimesNet)\]\[[slides](https://wuhaixu2016.github.io/pdf/ICLR2023_TimesNet.pdf)\]

---

## Transformer

- **Transformers in Time Series: A Survey**, _Wen et al._, IJCAI 2023. \[[paper](https://arxiv.org/abs/2202.07125)\]\[[code](https://github.com/qingsongedu/time-series-transformers-review)\]
- **Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting**, _Zhou et al._, AAAI 2021 Best paper. \[[paper](https://arxiv.org/abs/2012.07436)\]\[[code](https://github.com/zhouhaoyi/Informer2020)\]
- **Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting**, _Wu et al._, NeurIPS 2021. \[[paper](https://arxiv.org/abs/2106.13008)\]\[[code](https://github.com/thuml/Autoformer)\]\[[slides](https://wuhaixu2016.github.io/pdf/NeurIPS2021_Autoformer.pdf)\]
- **Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy**, _Xu et al._, ICLR 2022. \[[paper](https://arxiv.org/abs/2110.02642)\]\[[code](https://github.com/thuml/Anomaly-Transformer)\]\[[slides](https://wuhaixu2016.github.io/pdf/ICLR2022_Anomaly.pdf)\]
- **Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting**, _Liu et al._, NeurIPS 2022. \[[paper](https://arxiv.org/abs/2205.14415)\]\[[code](https://github.com/thuml/Nonstationary_Transformers)\]
- **iTransformer: Inverted Transformers Are Effective for Time Series Forecasting**, _Liu et al._, ICLR 2024 Spotlight. \[[paper](https://arxiv.org/abs/2310.06625)\]\[[code](https://github.com/thuml/iTransformer)\]
- **Pyraformer: Low-Complexity Pyramidal Attention for Long-Range Time Series Modeling and Forecasting**, _Liu et al._, ICLR 2022. \[[paper](https://openreview.net/forum?id=0EXmFzUn5I)\]\[[code](https://github.com/ant-research/Pyraformer)\]
- **FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting**, _Zhou et al._, ICML 2022. \[[paper](https://arxiv.org/abs/2201.12740)\]\[[code](https://github.com/MAZiqing/FEDformer)\]\[[DAMO-DI-ML](https://github.com/DAMO-DI-ML)\]
- PatchTST: **A Time Series is Worth 64 Words: Long-term Forecasting with Transformers**, _Nie et al._, ICLR 2023. \[[paper](https://arxiv.org/abs/2211.14730)\]\[[code](https://github.com/yuqinie98/PatchTST)\]
- **Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting**, _Zhang and Yan_, ICLR 2023.  \[[paper](https://openreview.net/forum?id=vSVLM2j9eie)\]\[[code](https://github.com/Thinklab-SJTU/Crossformer)\]
- **TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables**, _Wang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.19072)\]
- **UniTST: Effectively Modeling Inter-Series and Intra-Series Dependencies for Multivariate Time Series Forecasting**, _Liu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2406.04975)\]

---

## GNN

- **A Survey on Graph Neural Networks for Time Series: Forecasting, Classification, Imputation, and Anomaly Detection**, _Jin et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2307.03759)\]\[[code](https://github.com/KimMeen/Awesome-GNN4TS)\]
- **GPT-ST: Generative Pre-Training of Spatio-Temporal Graph Neural Networks**, _Li et al._, NeurIPS 2023. \[[paper](https://arxiv.org/abs/2311.04245)\]\[[code](https://github.com/HKUDS/GPT-ST)\]
- **MSGNet: Learning Multi-Scale Inter-Series Correlations for Multivariate Time Series Forecasting**, _Cai et al._, AAAI 2024. \[[paper](https://arxiv.org/abs/2401.00423)\]\[[code](https://github.com/YoZhibo/MSGNet)\]

---

## LLM Framework

- **Large Models for Time Series and Spatio-Temporal Data: A Survey and Outlook**, _Jin et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.10196)\]\[[code](https://github.com/qingsongedu/Awesome-TimeSeries-SpatioTemporal-LM-LLM)\]
- **Large Language Models for Time Series: A Survey**, _Zhang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.01801)\]\[[code](https://github.com/xiyuanzh/awesome-llm-time-series)\]
- **Large Language Models for Forecasting and Anomaly Detection: A Systematic Literature Review**, _Su et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.10350)\]

- **SimMTM: A Simple Pre-Training Framework for Masked Time-Series Modeling**, _Dong et al._, NeurIPS 2023 Spotlight. \[[paper](https://arxiv.org/abs/2302.00861)\]\[[code](https://github.com/thuml/SimMTM)\]
- **One Fits All:Power General Time Series Analysis by Pretrained LM**, _Zhou et al._, NeurIPS 2023. \[[paper](https://arxiv.org/abs/2302.11939)\]\[[code](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All)\]\[[AI-for-Time-Series-Papers-Tutorials-Surveys](https://github.com/DAMO-DI-ML/AI-for-Time-Series-Papers-Tutorials-Surveys)\]
- **Large Language Models Are Zero-Shot Time Series Forecasters**, _Gruver et al._, NeurIPS 2023. \[[paper](https://arxiv.org/abs/2310.07820)\]\[[code](https://github.com/ngruver/llmtime)\]
- **Lag-Llama: Towards Foundation Models for Time Series Forecasting**, _Rasul et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.08278)\]\[[code](https://github.com/time-series-foundation-models/lag-llama)\]\[[pytorch-transformer-ts](https://github.com/kashif/pytorch-transformer-ts)\]
- TimesFM: **A decoder-only foundation model for time-series forecasting**, _Das et al._, ICML 2024. \[[paper](https://arxiv.org/abs/2310.10688)\]\[[code](https://github.com/google-research/timesfm)\]
- **TimeGPT-1**, _Garza et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2310.03589)\]\[[nixtla](https://github.com/Nixtla/nixtla)\]
- **Time-LLM: Time Series Forecasting by Reprogramming Large Language Models**, _Jin et al._, ICLR 2024. \[[paper](https://arxiv.org/abs/2310.01728)\]\[[code](https://github.com/KimMeen/Time-LLM)\]
- **AutoTimes: Autoregressive Time Series Forecasters via Large Language Models**, _Liu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.02370)\]\[[code](https://github.com/thuml/AutoTimes)\]
- **Timer: Transformers for Time Series Analysis at Scale**, _Liu et al._, ICML 2024. \[[paper](https://arxiv.org/abs/2402.02368)\]\[[code](https://github.com/thuml/Large-Time-Series-Model)\]\[[website](https://thuml.github.io/timer)\]
- **MOMENT: A Family of Open Time-series Foundation Models**, _Goswami et al._, ICML 2024. \[[paper](https://arxiv.org/abs/2402.03885)\]\[[code](https://github.com/moment-timeseries-foundation-model/moment)\]
- **Unified Training of Universal Time Series Forecasting Transformers**, _Woo et al._, ICML 2024. \[[paper](https://arxiv.org/abs/2402.02592)\]\[[code](https://github.com/SalesforceAIResearch/uni2ts)\]
- **Multi-Patch Prediction: Adapting LLMs for Time Series Representation Learning**, _Bian et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2402.04852)\]
- **UniTS: Building a Unified Time Series Model**, _Gao et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.00131)\]\[[code](https://github.com/mims-harvard/UniTS)\]
- **Chronos: Learning the Language of Time Series**, _Ansari et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.07815)\]\[[code](https://github.com/amazon-science/chronos-forecasting)\]
- **Large language models can be zero-shot anomaly detectors for time series**, _Alnegheimish et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.14755)\]
- **Foundation Models for Time Series Analysis: A Tutorial and Survey**, _Liang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.14735)\]\[[granite-tsfm](https://github.com/ibm-granite/granite-tsfm)\]

---
## Diffusion Model
- **Diffusion-TS: Interpretable Diffusion for General Time Series Generation**, _Yuan and Qiao_, ICLR 2024. \[[paper](https://arxiv.org/abs/2403.01742)\]\[[code](https://github.com/Y-debug-sys/Diffusion-TS)\]
- **A Survey on Diffusion Models for Time Series and Spatio-Temporal Data**, _Yang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2404.18886)\]\[[code](https://github.com/yyysjz1997/Awesome-TimeSeries-SpatioTemporal-Diffusion-Model)\]

---

## Benchmark and Dataset
- **TSPP: A Unified Benchmarking Tool for Time-series Forecasting**, _Bączek et al._, arxiv 2023. \[[paper](https://arxiv.org/abs/2312.17100)\]\[[code](https://github.com/NVIDIA/DeepLearningExamples/tree/master/Tools/PyTorch/TimeSeriesPredictionPlatform)\]
- **TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods**, _Qiu et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2403.20150)\]\[[code](https://github.com/decisionintelligence/TFB)\]
- **A Survey of Generative Techniques for Spatial-Temporal Data Mining**, _Zhang et al._, arxiv 2024. \[[paper](https://arxiv.org/abs/2405.09592)\]
  
- \[[multivariate-time-series-data](https://github.com/laiguokun/multivariate-time-series-data)\]\[[ETDataset](https://github.com/zhouhaoyi/ETDataset)\]\[[Awesome-TimeSeries-SpatioTemporal-Diffusion-Model](https://github.com/yyysjz1997/Awesome-TimeSeries-SpatioTemporal-Diffusion-Model)\]

---

## Repositories

- \[[Time-Series-Library](https://github.com/thuml/Time-Series-Library)\]
- \[[time-series-transformers-review](https://github.com/qingsongedu/time-series-transformers-review)\]\[[awesome-AI-for-time-series-papers](https://github.com/qingsongedu/awesome-AI-for-time-series-papers)\]\[[Awesome-TimeSeries-SpatioTemporal-LM-LLM](https://github.com/qingsongedu/Awesome-TimeSeries-SpatioTemporal-LM-LLM)\]\[[TSFpaper](https://github.com/ddz16/TSFpaper)\]\[[deep-learning-time-series](https://github.com/Alro10/deep-learning-time-series)\]
- \[[statsforecast](https://github.com/Nixtla/statsforecast)\]\[[neuralforecast](https://github.com/Nixtla/neuralforecast)\]\[[gluonts](https://github.com/awslabs/gluonts)\]\[[Merlion](https://github.com/salesforce/Merlion)\]\[[pytorch-forecasting](https://github.com/jdb78/pytorch-forecasting)\]\[[tsai](https://github.com/timeseriesAI/tsai)\]
- \[[AIAlpha](https://github.com/VivekPa/AIAlpha)\]
- \[[prophet](https://github.com/facebook/prophet)\]\[[Kats](https://github.com/facebookresearch/Kats)\]\[[tsfresh](https://github.com/blue-yonder/tsfresh)\]\[[sktime](https://github.com/sktime/sktime)\]\[[darts](https://github.com/unit8co/darts)\]\[[tslearn](https://github.com/tslearn-team/tslearn)\]\[[pyflux](https://github.com/RJT1990/pyflux)\]


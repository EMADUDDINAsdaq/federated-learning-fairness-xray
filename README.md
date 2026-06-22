#EVALUATION OF FEDERATED LEARNING AGGREGATION STRATEGIES UNDER DATA IMBALANCE ON CHEST X-RAY IMAGES: ADDRESSING CLIENT AND DEMOGRAPHIC FAIRNESS
dataset link - 
https://www.kaggle.com/datasets/nih-chest-xrays/data/


Author: Emaduddin Asdaq Syed Mohammed




Research Question

Which federated learning aggregation strategy — FedAvg, q-FedAvg, or GIFAIR-FL — best reduces both inter-client AUC variance across simulated hospital clients AND intra-client demographic FNR gaps across patient age and sex subgroups when applied to NIH ChestX-ray14 chest X-ray classification?


Original Contribution

First empirical comparison of FedAvg, q-FedAvg, and GIFAIR-FL on NIH ChestX-ray14 with real patient demographic metadata (age and sex), measuring both inter-client performance fairness and intra-client demographic fairness simultaneously.


Dataset


NIH ChestX-ray14 — Wang et al. 2017, IEEE CVPR
112,104 chest X-ray images after cleaning
Metadata: patient age, sex, 15 pathology labels
Binary label: 1 = any pathology, 0 = No Finding
Partitioned across 5 simulated hospital clients using Dirichlet Dir(α=0.5)
dataset link - 
https://www.kaggle.com/datasets/nih-chest-xrays/data/


Methods Compared

MethodCitationKey MechanismFedAvgMcMahan et al. 2017Weighted average by dataset sizeq-FedAvgLi et al. 2020Gradient reweighting by loss^q — struggling clients get more influenceGIFAIR-FLYue et al. 2022Fairness penalty inside local gradient — plain average aggregation


Hospital Client Statistics (Dirichlet Dir α=0.5, seed=42)

ClientImagesPathology%Hospital_A61,50559.9%Hospital_B10,65599.7%Hospital_C34,9368.2%Hospital_D1,49569.1%Hospital_E3,51310.0%

Variance across clients: 0.125535 — strong non-IID confirmed.


Model and Training


Model: ResNet-18 pretrained ImageNet — Zech et al. 2018
Loss: BCEWithLogitsLoss (binary cross-entropy)
Optimiser: Adam lr=1e-4
Rounds: 10 | Epochs/round: 3 | Batch size: 256
GPU: Google Colab L4



Evaluation Metrics


AUC — Area Under ROC Curve — per client and per demographic subgroup
FNR — False Negative Rate = FN/(FN+TP) — clinical underdiagnosis measure
Subgroups: Sex (M/F) | Age groups (0-20, 20-40, 40-60, 60-80, 80+)
Citation: Seyyed-Kalantari et al. 2021 Nature Medicine



Implementation Notes

The FL simulation uses Flower (flwr) client and strategy abstractions — fl.client.NumPyClient, fl.server.strategy.Strategy — following Beutel et al. 2020. Due to a known incompatibility between flwr 1.31.0 and Ray on Python 3.12 in the Colab environment, the simulation runner was implemented directly using PyTorch while retaining Flower's client and strategy class structures.


Preliminary Results (10% sample, 5 rounds, batch=32)

ClientFedAvg AUCq-FedAvg AUCGIFAIR AUCHospital_A0.95710.85830.9243Hospital_B0.30120.61700.1729Hospital_C0.88680.86190.8714Hospital_D0.67350.90360.7166Hospital_E0.70170.90970.7516

AUC Variance: FedAvg=0.052 | q-FedAvg=0.0118 (77.4% reduction) | GIFAIR=0.0719

FNR Sex Gap: FedAvg=0.0847 | q-FedAvg=0.0944 | GIFAIR=0.0083





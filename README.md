# fedAVG_fedProx_fedBN_test
## [Xiaoxiao Li et al., 2021](https://openreview.net/pdf?id=6YEQUn0QICG) proposed FedBN, a federated learning approach for non-IID features via local batch normalization.

**Benchmark(Digits)**
- Please download our pre-processed datasets [here](https://drive.google.com/file/d/1moBE_ASD5vIOaU8ZHm_Nsj0KAfX5T0Sf/view?usp=sharing](https://drive.google.com/u/0/uc?id=1moBE_ASD5vIOaU8ZHm_Nsj0KAfX5T0Sf&export=download&confirm=t&uuid=fece15d8-45eb-467a-97d5-542a71fd7f3c&at=ALgDtsyllrEea3CehaYWzXuwlv3u:1678400117046)), put under `data/` directory and perform following commands:
    ```bash
    cd ./data
    unzip digit_dataset.zip
    ```
 ### Train

use following command.
- **--mode** specify federated learning strategy, option: fedavg | fedprox | fedbn 
```bash
cd federated
# benchmark experiment
python fed_digits.py --mode fedbn
```
**Next (stilll working on this)**
 - Benchmarking some MedMnist Datasets
 

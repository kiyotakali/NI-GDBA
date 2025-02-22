


## Dataset
- `AIDS`,`NCI1`,`PROTEINS_full`,`DD`, `ENZYMES` ,  `COLORS-3`


## GNN Model
We consider the most widely studied GNN models:
- **GCN**.
- **GAT**.
- **GraphSAGE**.


## Attack results
The baseline attack can be realized by this code.
```python
python run_baseline1.py --dataset NCI1 \
                         --config ./Graph_level_Models/configs/TUS/TUs_graph_classification_GCN_NCI1_100k.json \
                         --is_iid iid\
                         --num_workers 10\
                         --num_mali 3\
                         --epoch_backdoor 0\
                         --frac_of_avg 0.1\
                         --trigger_type ba\
                         --trigger_position random\
                         --poisoning_intensity 0.1\
                         --filename ./checkpoints/Graph \
                         --device_id 0
```

Our NI-GDBA can be realized by:
```python
python run_our.py --dataset NCI1 \
                         --config ./Graph_level_Models/configs/TUS/TUs_graph_classification_GCN_NCI1_100k.json \
                         --is_iid iid\
                         --num_workers 10\
                         --num_mali 3\
                         --epoch_backdoor 0\
                         --frac_of_avg 0.1\
                         --trigger_type renyi\
                         --trigger_position cluster\
                         --poisoning_intensity 0.1\
                         --filename ./checkpoints/Graph \
                         --device_id 2
```


## Preparation




## Install metis


```python
wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz
gunzip metis-5.1.0.tar.gz
tar -xvf metis-5.1.0.tar
rm metis-5.1.0.tar
cd metis-5.1.0
make config shared=1
make install
export METIS_DLL=/usr/local/lib/libmetis.so

pip3 install metis-python
```

## Environment settings and libraries we used in our experiments

This project is tested under the following environment settings:

- appdirs==1.4.4
- brotlipy==0.7.0
- cachetools==5.3.0
- certifi==2022.12.7
- cffi==1.15.0
- chardet==5.1.0
- charset-normalizer==3.0.1
- click==8.1.3
- contourpy==1.0.7
- cryptography==38.0.4
- cycler==0.11.0
- Cython==0.29.33
- dgl==1.0.1+cu116
- docker-pycreds==0.4.0
- Flask==2.2.3
- fonttools==4.39.0
- gitdb==4.0.10
- GitPython==3.1.31
- hdbscan==0.8.28
- idna==3.4
- importlib-metadata==6.0.0
- importlib-resources==5.12.0
- itsdangerous==2.1.2
- Jinja2==3.1.2
- joblib==1.1.0
- kiwisolver==1.4.4
- MarkupSafe==2.1.2
- matplotlib==3.7.1
- mkl-fft==1.3.1
- mkl-random==1.2.2
- mkl-service==2.4.0
- networkx==3.0
- numpy==1.24.2
- nvidia-cublas-cu11==11.10.3.66
- nvidia-cuda-nvrtc-cu11==11.7.99
- nvidia-cuda-runtime-cu11==11.7.99
- nvidia-cudnn-cu11==8.5.0.96
- nvidia-ml-py==11.525.84
- nvitop==1.0.0
- packaging==23.0
- pathtools==0.1.2
- Pillow==9.4.0
- pip==22.3.1
- protobuf==4.22.3
- psutil==5.9.4
- pycparser==2.21
- pyg-lib==0.1.0+pt113cu117
- pyOpenSSL==22.0.0
- pyparsing==3.0.9
- PySocks==1.7.1
- python-dateutil==2.8.2
- python-louvain==0.16
- PyYAML==6.0
- requests==2.28.2
- scikit-learn==1.1.3
- scikit-learn-extra==0.2.0
- scipy==1.10.1
- sentry-sdk==1.20.0
- setproctitle==1.3.2
- setuptools==65.6.3
- six==1.16.0
- smmap==5.0.0
- termcolor==2.2.0
- threadpoolctl==3.1.0
- torch==1.13.1
- torch-cluster==1.6.0+pt113cu117
- torch-geometric==2.2.0
- torch-scatter==2.1.0+pt113cu117
- torch-sparse==0.6.16+pt113cu117
- torch-spline-conv==1.2.1+pt113cu117
- torchvision==0.14.1
- tqdm==4.64.1
- typing_extensions==4.5.0
- urllib3==1.26.14
- wandb==0.15.2
- Werkzeug==2.2.3
- wheel==0.38.4
- zipp==3.15.0

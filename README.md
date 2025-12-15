This repository is a customized version of the ScalarLM framework, tailored for the [AI Telco Troubleshooting Challenge](https://aiforgood.itu.int/ai-telco-troubleshooting-challenge/) (TeleLogs) hosted by [GSMA](https://www.linkedin.com/company/gsma/), happening from November 2025 to March 2026. TeleLogs is a world-wide competition focused on training language models for network root cause analysis. The main objective of this repository is to provide guidelines and resources for post-training language models relevant to the challenge. 

ScalarLM is a GPU-agnostic, fully open-source, CC-0 licensed platform for inference, training, and deployment. ScalarLM builds on top of the vLLM inference engine, the Megatron-LM training framework, and the HuggingFace model hub. To learn more about ScalarLM, please visit [ScalarLM](https://www.scalarlm.com/).

For this phase of the competition, we will utilize TensorWave Cloud as our primary GPU provider, leveraging AMD MI300X and MI355X GPUs for large-scale training and inference as part of the challenge. However, this repository remains compatible with both NVIDIA and AMD GPUs.

Please follow the steps below to set up the environemnt and submit a job:

- Create a virtual environment: 
    - python3 -m venv scalarlm_env
- Activate the environment: 
    - source scalarlm_env/bin/activate
- Install scalarlm and yaml libraries: 
    - pip3 install scalarlm pyyaml
- Connect to a certain endpoint for training: 
    - export SCALARLM_API_URL={a url that will be provided to you}
- Submit a job for training: 
    - python3 train.py

Once you submit a job, you will receive a model_id. You can use this model_id to view the logs by running the following command: 

    scalarlm logs --model={model_id}

If you have any technical questions, please feel free to reach out to farbod.tavakkoli@att.com or farbodtavakoli@gmail.com

Acknowledgement: [AT&T](https://www.linkedin.com/company/att/posts/?feedView=all), [AMD](https://www.linkedin.com/company/amd/), [TensorWave](https://www.linkedin.com/company/tensorwave/), and [RelationalAI](https://www.linkedin.com/company/relationalai/posts/?feedView=all).

Authors: [Farbod Tavakkoli](https://www.linkedin.com/in/farbodtavakkoli/),  [Gregory Diamos](https://www.linkedin.com/in/gregory-diamos-1a8b9083/),  [Jorden Terrazas](https://www.linkedin.com/in/jorden-terrazas-4a440714a/),  [Roderic Paulk](https://www.linkedin.com/in/roderic-paulk-64a30718/)

Last Update Date: December 15, 2025.

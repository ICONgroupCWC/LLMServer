This repository holds code for LLM server for automated federated learning for user prompts. To run the LLM server install below dependencies.

1. [llama_cpp](https://github.com/ggerganov/llama.cpp)
2. [flask](https://flask.palletsprojects.com/en/stable/installation/)
3. [openai](https://pypi.org/project/openai/)

To run the server run following command *python inference_server.py*

The LLM server here is for supporting requests from the FL server implemented [here](https://github.com/ICONgroupCWC/FedLBE). The standalone code for running NAS/HPO can be found the this [repo](https://github.com/MChamith/AutomatedNASFL).



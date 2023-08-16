# miCSE: mutual information Contrastive Sentence Embedding for Low-shot Sentence Embeddings
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/arXiv-2109.05105-29d634.svg)](https://arxiv.org/abs/2211.04928)
[![REUSE status](https://api.reuse.software/badge/github.com/SAP-samples/acl2023-micse)](https://api.reuse.software/info/github.com/SAP-samples/acl20230-micse)


#### News
- **Source code will be provided shortly**
- **08/16/2023:** :confetti_ball: Training data provided :tada:


## Description
This repository **will contain** the source code for our paper [**miCSE: mutual information Contrastive Sentence Embedding for Low-shot Sentence Embeddings**](https://arxiv.org/abs/2211.04928) to be presented at [ACL2023](https://2023.aclweb.org/). 

### Abstract
![Schematic Illustration of miCSE](https://github.com/SAP-samples/acl2023-micse/blob/96c833426b637fc35ca071dc62dfd89e96aee08c/images/ami_pipeline.png)
This paper presents **miCSE**, a mutual information-based Contrastive learning framework that significantly advances the state-of-the-art in few-shot sentence embedding.The proposed approach imposes alignment between the attention pattern of different views during contrastive learning. Learning sentence embeddings with miCSE entails enforcing the structural consistency across augmented views for every single sentence, making contrastive self-supervised learning more sample efficient. As a result, the proposed approach shows strong performance in the few-shot learning domain. While it achieves superior results compared to state-of-the-art methods on multiple benchmarks in few-shot learning, it is comparable in the full-shot scenario.
This study opens up avenues for efficient self-supervised learning methods that are more robust than current contrastive methods for sentence embedding.


## Language Models

Language models trained for which the performance is reported in the paper are available at the [Huggingface Model Repository](https://huggingface.co/models):
 - [BERT-base-uncased: sap-ai-research/miCSE](https://huggingface.co/sap-ai-research/miCSE)

```shell
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("sap-ai-research/miCSE")

model = AutoModel.from_pretrained("sap-ai-research/miCSE")
```

## Data
The model was trained on a random collection of **English** sentences from Wikipedia: [Training data file](https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/wiki1m_for_simcse.txt)
Training data consists of data splits of different sizes (from 10% to 0.0064%) of the [SimCSE](https://github.com/princeton-nlp/SimCSE) training corpus. Each split size comprises 5 different files, each file created with a different seed. To download the data:

```shell
cd data
sh download.sh
```

#### Authors:
 - [Tassilo Klein](https://tjklein.github.io/)
 - [Moin Nabi](https://moinnabi.github.io/)

## Requirements
- [Python](https://www.python.org/) (version 3.6 or later)
- [PyTorch](https://pytorch.org/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)


## Citations
If you use this code in your research or want to refer to our work, please cite:

```
@inproceedings{klein-nabi-2023-micse,
    title = "mi{CSE}: Mutual Information Contrastive Learning for Low-shot Sentence Embeddings",
    author = "Klein, Tassilo  and
      Nabi, Moin",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.339",
    pages = "6159--6177",
    abstract = "This paper presents miCSE, a mutual information-based contrastive learning framework that significantly advances the state-of-the-art in few-shot sentence embedding.The proposed approach imposes alignment between the attention pattern of different views during contrastive learning. Learning sentence embeddings with miCSE entails enforcing the structural consistency across augmented views for every sentence, making contrastive self-supervised learning more sample efficient. As a result, the proposed approach shows strong performance in the few-shot learning domain. While it achieves superior results compared to state-of-the-art methods on multiple benchmarks in few-shot learning, it is comparable in the full-shot scenario. This study opens up avenues for efficient self-supervised learning methods that are more robust than current contrastive methods for sentence embedding.",
}
```

## How to obtain support
[Create an issue](https://github.com/SAP-samples/<repository-name>/issues) in this repository if you find a bug or have questions about the content.
 
For additional support, [ask a question in SAP Community](https://answers.sap.com/questions/ask.html).

## Contributing
If you wish to contribute code, offer fixes or improvements, please send a pull request. Due to legal reasons, contributors will be asked to accept a DCO when they create the first pull request to this project. This happens in an automated fashion during the submission process. SAP uses [the standard DCO text of the Linux Foundation](https://developercertificate.org/).

## License
Copyright (c) 2023 SAP SE or an SAP affiliate company. All rights reserved. This project is licensed under the Apache Software License, version 2.0 except as noted otherwise in the [LICENSE](LICENSES/Apache-2.0.txt) file.

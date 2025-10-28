# KGFlow





## Knowledge Graph



KGFlow uses PrimeKG, Hetionet, and UMLS as its knowledge bases.

You can download them from the following links:

PrimeKG: https://zitniklab.hms.harvard.edu/projects/PrimeKG/

Hetionet: https://het.io/

UMLS: https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html



## Dataset



We used five multiple-choice medical QA datasets to evaluate our proposed model: MMLU-Med, MedQA-US, MedMCQA, PubMedQA, and BioASQ-Y/N.

You can find them in the `Dataset` folder of this repository.



## Usage



To use KGFlow, follow these steps:



### 1. Install the environment



Bash

```
conda env create -f environment.yml
```



### 2. Prepare Neo4j and LLM



You can download and install Neo4j from https://neo4j.com/. The Docker version is recommended.

The LLM can be downloaded from https://huggingface.co/, or it will be downloaded automatically when the script runs.



### 3. Run the main KGFlow program



Bash

```
python kgflow.py --model model_dir --umls_url bolt://host:port --umls_username username --umls_password password --primekg_url bolt://host:port --primekg_username username --primekg_password password --hetionet_url bolt://host:port --hetionet_username username --hetionet_password password
```
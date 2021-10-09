# [NAACL'21] EventPlus: A Temporal Event Understanding Pipeline

This is the codebase for the system demo EventPlus: A Temporal Event Understanding Pipeline in NAACL 2021.

Please refer to our paper for details. [[PDF]](https://www.aclweb.org/anthology/2021.naacl-demos.7.pdf) [[Talk]](https://youtu.be/KPXpKeVIuag) [[Demo]](https://kairos-event.isi.edu/)

## Quick Start

0 - Clone the codebase with all submodules

```
git clone --recurse-submodules https://github.com/PlusLabNLP/EventPlus.git
# or use following commands
git clone https://github.com/PlusLabNLP/EventPlus.git
git submodule init
git submodule update
```

1 - Environment Installation

Change prefix (last line) of `env.yml` to fit your path, then run

```
conda env create -f env.yml
conda activate event-pipeline
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_ner_jnlpba_md-0.2.4.tar.gz
python -m spacy download en_core_web_sm
pip install git+https://github.com/hltcoe/PredPatt.git
```

2 - Download trained model for components

For `component/BETTER` module, download the trained model [[Link]](https://drive.google.com/file/d/19_W6azeG5KRQxLDICswqwIFX0QOjxh_L/view?usp=sharing), unzip and place it under `component/BETTER/joint/worked_model_ace`. Also:

For `component/Duration` module, download `scripts` zip file [[Link]](https://drive.google.com/file/d/1s1uLcQjjFdfcto3BZ3aRi8pPzLf9KELe/view?usp=sharing), unzip and place it under `component/Duration/scripts`.

For `component/NegationDetection` module, download the trained model [[Link]](https://drive.google.com/file/d/1FLAHrWy3eF23Kb7Ql4k_f1a5lCQ5m1L0/view?usp=sharing), unzip and place is under `component/NegationDetection/models`

3 - In background: Run REST API for event duration detection module for faster processing
```
(optional) tmux new -s duration_rest_api
conda activate event-pipeline
cd component/REST_service
python main.py
(optional) exit tmux window
```

4 - Application 1: Raw Text Annotation. The input is a multiple line raw text file, and the output pickle and json file will be saved to designated paths
```
cd YOUR_PREFERRED_PATH/project
python APIs/test_on_raw_text.py -data YOUR_RAW_TEXT_FILE -save_path SAVE_PICKLE_PATH -save_path_json SAVE_JSON_PATH -negation_detection
```

5 - Application 2: Web App for Interaction and Visualization. A web app will be started and user can input a piece of text and get annotation result and visualization.
```
cd YOUR_PREFERRED_PATH/project
tmux new -s serve
python manage.py runserver 8080
```

## Components

The code for data processing and incorporating different components is in `project/APIs/main.py`. Please refer to README file of each component for more details about training and inference.

1- Event Extraction on ACE Ontology: `component/BETTER`
 
2- Joint Event Trigger and Temporal Relation Extraction: `component/TempRel` for inference, [this codebase](https://github.com/rujunhan/EMNLP-2019) for training

3- Event Duration Detection: `component/Duration`

4- Negation and Speculation Cue Detection and Scope Resolution: `component/NegationDetection`

5- Biomedical Event Extraction: `component/BioMedEventEx` for inference, [this codebase](https://github.com/PlusLabNLP/GEANet-BioMed-Event-Extraction) for training

## Quick Start with ISI shared NAS

If you are using the system on a machine with access to ISI shared NAS, you could directly activate environment and copy the code and start using it right away!

```
# 1 - Environment Installation: Activate existing environment
conda activate /nas/home/mingyuma/miniconda3/envs/event-pipeline-dev

# 2 - Prepare Components (Submodules): Copy the whole codebase
cp -R /nas/home/mingyuma/event-pipeline/event-pipeline-dev YOUR_PREFERRED_PATH

# 3 - In background: Run REST API for event duration detection module for faster processing
(optional) tmux new -s duration_rest_api
conda activate /nas/home/mingyuma/miniconda3/envs/event-pipeline-dev
cd component/REST_service
python main.py
(optional) exit tmux window

# To use it for raw text annotation or web app, please follow step 4 and 5 in quick start section.
```

## Deployment as Web Service

Here are instruction of how to deploy the web application on an server

### Set up web server

```
pip install uwsgi
```

If you met errors like `error while loading shared libraries libssl.so.1.1`, reference [this link](https://www.bswen.com/2018/11/others-Openssl-version-cause-error-when-loading-shared-libraries-libssl.so.1.1.html) and do the following

```
export LD_LIBRARY_PATH=/nas/home/mingyuma/miniconda3/envs/event-pipeline/lib:$LD_LIBRARY_PATH
```

### Server port setting

External port: 443 (for HTTPS)

Django will forward traffic from 443 port to internal 8080 port

Internal port
* 8080: run Django main process
* 17000: run service for duration (if we run a REST API for duration module, but now the newer version doesn't need such a separate service)

## Citation

```
@inproceedings{ma-etal-2021-eventplus,
    title = "{E}vent{P}lus: A Temporal Event Understanding Pipeline",
    author = "Ma, Mingyu Derek  and
      Sun, Jiao  and
      Yang, Mu  and
      Huang, Kung-Hsiang  and
      Wen, Nuan  and
      Singh, Shikhar  and
      Han, Rujun  and
      Peng, Nanyun",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies: Demonstrations",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-demos.7",
    pages = "56--65",
    abstract = "We present EventPlus, a temporal event understanding pipeline that integrates various state-of-the-art event understanding components including event trigger and type detection, event argument detection, event duration and temporal relation extraction. Event information, especially event temporal knowledge, is a type of common sense knowledge that helps people understand how stories evolve and provides predictive hints for future events. EventPlus as the first comprehensive temporal event understanding pipeline provides a convenient tool for users to quickly obtain annotations about events and their temporal information for any user-provided document. Furthermore, we show EventPlus can be easily adapted to other domains (e.g., biomedical domain). We make EventPlus publicly available to facilitate event-related information extraction and downstream applications.",
}
```
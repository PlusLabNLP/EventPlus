# UDS-T

Event Duration Baselines on UDS-T

> This repo provides baseline models and evaluations for 
time-duration classification, on USD-T dataset.

---

## Environment Setup
```
conda create -n event_dur
conda install pip
pip install git+https://github.com/hltcoe/PredPatt.git
pip install -r requirements.txt
```



## Inference API

See requirements.txt for required dependencies. <br><br>
Example for performing duration model inference.
```python
from inference_api import predict_duration_elmo

# events_json = EventsModel(...)
# events = json.loads(events_json)  # Parse JSON string
out_json_str = predict_duration_elmo(events)
```

Output JSON structure:
```json5
[
  {
    'duration': 'days',
    'pred_text': 'meeting',
    'pred_idx': 13,
    'sentence': 'There was a ...' 
  },
]
```


---
## Models


- ELMo-MLP Baseline 
    <br>
    - Under eval mode (torch.no_grad), consumes ~ 1.3GB GPU-RAM, for batch-size=1

<br>

- BERT/RoBERTa Baseline


<br>

<br>

---

## Training

Run the following script for training:



```bash
$ python3 main.py \
--mode train
```



<br>
---

*TO-DOs*

- [ ] Add BERT baseline



## References
[1]  [Fine-Grained Temporal Relation Extraction](https://www.aclweb.org/anthology/P19-1280/) <br>
[2]  []() <br>

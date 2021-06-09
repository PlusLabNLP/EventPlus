# Model for BETTER Project 
## Event Extraction system API

### Download pretrained models
The pretrained models are [here](https://drive.google.com/file/d/19_W6azeG5KRQxLDICswqwIFX0QOjxh_L/view?usp=sharing). Download the models and unzip it. There should be a `worked_model_ace` folder under `joint`.

### Run code

```
python event_pipeline_demo.py
```

For the core of calling the event extraction system, see line 127-137 in `event_pipeline_demo.py`. The expected output should be 
```
[{'tokens': ['Orders', 'went', 'out', 'today', 'to', 'deploy', '17,000', 'U.S.', 'Army', 'soldiers', 'in', 'the', 'Persian', 'Gulf', 'region', '.'], 'events': [{'event_type': 'Movement:Transport', 'triggers': [{'event_type': 'Movement:Transport', 'text': 'deploy', 'start_token': 5, 'end_token': 5}], 'arguments': [{'role': 'Artifact', 'text': 'soldiers', 'start_token': 9, 'end_token': 9}, {'role': 'Destination', 'text': 'region', 'start_token': 14, 'end_token': 14}]}], 'ner': [[7, 7, 'GPE'], [8, 8, 'ORG'], [9, 9, 'PER'], [12, 13, 'LOC'], [14, 14, 'LOC']]}]
```

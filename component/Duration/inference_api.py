import torch
import sys
import os
import json
import pandas as pd
from torch.utils.data import DataLoader
from os.path import dirname, abspath
parent_dir = dirname(dirname(abspath(__file__)))
sys.path.insert(0, parent_dir)
from utils_duration import str2bool, compute_predictions, idx2label
from preprocess import TempEveDataset
# from allennlp.commands.elmo import ElmoEmbedder
from allennlp.modules.elmo import Elmo
from .scripts.src.factslab.factslab.pytorch.temporalmodule import TemporalModel

class DurationAPI:
    def __init__(self, base_dir = '.', gpu_id=-1):
        """
        :param int gpu_id: cuda device id (optional); default - cpu
        """
        self.base_dir = base_dir
        device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() and gpu_id != -1 else 'cpu')

        # Model Configs
        options_file = os.path.join(base_dir, "./scripts/elmo_files/elmo_2x4096_512_2048cnn_2xhighway_options.json")
        weight_file = os.path.join(base_dir, "./scripts/elmo_files/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5")

        model_ckpt = os.path.join(base_dir, "./model_ckpt/model_param_param_param_1_0_128_128_0_0_0_0_0.0_0.5_relu_1.pth")
        file_name = model_ckpt.split('/')[-1]

        tokens = file_name.split("_")
        eventatt = tokens[1]
        duratt = tokens[2]
        relatt = tokens[3]
        concat_fine_to_dur = str2bool(tokens[-8])
        concat_dur_to_fine = str2bool(tokens[-7])
        fine_2_dur = str2bool(tokens[-6])
        dur_2_fine = str2bool(tokens[-5])
        weight = float(tokens[-4])
        drop = float(tokens[-3])
        activ = tokens[-2]
        bino_bool = str2bool(tokens[-1].split(".")[0])

        # coarse_size = int(tokens[-1].split(".")[0])

        print("Eventatt: {}, Duratt: {}, Relatt: {}, Dropout: {}, Activation: {}, Binomial: {}, "
            "concat_fine2dur: {}, concat_dur2fine:{}, fine_to_dur: {}, dur_to_fine: {} \n"
            .format(eventatt, duratt, relatt, drop, activ, bino_bool,
                    concat_fine_to_dur, concat_dur_to_fine, fine_2_dur, dur_2_fine))

        self.batch_size = 1
        self.num_workers = 1

        # Model
        self.model = TemporalModel(embedding_size=1024, duration_distr=bino_bool,
                            # elmo_class=
                            mlp_dropout=drop, mlp_activation=activ, tune_embed_size=256, event_attention=eventatt,
                            dur_attention=duratt, rel_attention=relatt, concat_fine_to_dur=concat_fine_to_dur,
                            concat_dur_to_fine=concat_dur_to_fine, fine_to_dur=fine_2_dur, dur_to_fine=dur_2_fine,
                            fine_squash=True, baseline=False, dur_MLP_sizes=[128], fine_MLP_sizes=[128],
                            dur_output_size=11, fine_output_size=4, device=device)
        
        self.model.to(device)

        # Load model weights
        checkpoint = torch.load(model_ckpt, map_location=device)
        self.model.load_state_dict(checkpoint)
        self.model.elmo_class = Elmo(options_file, weight_file, num_output_representations=3)
    
    def pred(self, events):
        """
        Model inference for ELMo baseline, given Events JSON

        :param list[dict] events: list of sentences and extracted event-triggers (within dict)
        :return: json containing event-duration as list of dict
        :rtype: str
        """
        # Dataloader
        test_dataset = TempEveDataset(events)

        test_loader = DataLoader(test_dataset, self.batch_size, num_workers=self.num_workers, drop_last=False)

        # Inference
        outputs = compute_predictions(self.model, test_loader)

        # DataFrame
        df_out = pd.DataFrame(outputs)
        df_out = df_out[['p1_dur', 'root_text', 'root_idx', 'sentence']]

        df_out.rename(columns={'p1_dur': 'duration',
                            'root_text': 'pred_text',
                            'root_idx': 'pred_idx'}, inplace=True)

        # Map duration index to label
        df_out['duration'] = df_out['duration'].apply(lambda idx: idx2label[idx])

        json_str = df_out.to_json(orient='records')

        # Parse json string to List[dict]
        json_obj = json.loads(json_str)
        return json_obj


if __name__ == '__main__':
    # Input
    json_file = './Mu_test_data/dev_tbd.pred.json'

    # Read json file
    events_input = json.load(open(json_file))

    # For demo, input is obtained from Mu's Event model,
    # thus first decode the json string as follows:
    # events_input = json.loads(events_json_str)   # str --> List[dict]
    # result = predict_duration_elmo(events_input, gpu_id=0)
    events_input = events_input[:2]
    print(events_input)

    api = DurationAPI()
    result = api.pred(events_input)

    print(result)

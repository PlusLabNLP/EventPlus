from scripts.utils import *
from scripts.timelinemodule import TimelineModel
import argparse
import warnings


warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-doc", "--docpath",
                        help="Path of the document file",
                        type=str,
                        default="")

    parser.add_argument("-gpu", "--gpunumber",
                        help="Which gpu to use",
                        type=int,
                        default=1)

    parser.add_argument("-out", "--outpath",
                        help="Path of the output folder",
                        type=str,
                        default="")

    args = parser.parse_args()

    ## Dependency Graph object
    filename = args.docpath.split("/")[-1]
    structures = get_structs(args.docpath)
    print("\n###########   Parsing Conllu through PredPatt    ###########")

    ## Sentences
    struct_dict = extract_struct_dicts(structures)

    ## A dataframe after processing the file through PredPatt and extracting
    ## roots and spans of each predicate.
    df = extract_dataframe(args.docpath, structures)

    ## Correct pred2_tokens as per the concatenated sentence
    df['pred2_token_mod'] = df.apply(lambda row: correct_pred2_tokens(row, struct_dict), axis=1)
    df['pred2_root_token_mod'] = df.apply(lambda row: correct_pred2_root(row, struct_dict), axis=1)
    # Convert tokens into list of numbers
    df['pred1_token_span'] = df['pred1_token'].map(lambda x: [int(y) for y in x.split("_")])
    df['pred2_token_span'] = df['pred2_token_mod'].map(lambda x: [int(y) for y in x.split("_")])

    ## Extract X for model predictions
    X = extract_X(df)

    ## Load the best model
    squashed = True
    baseline = False
    loss_confidence = True
    cuda_device_num = args.gpunumber
    cuda_device_str = "cuda:" + str(cuda_device_num)
    model_path = "../model/"
    file_path = "model_param_param_param_1_0_128_128_0_0_0_0_0.0_0.5_relu_1.pth"

    tokens = file_path.split("_")
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
    print("\n###########   Predicting Relative Timelines    ###########")
    print("\nRelative Temporal Model configurations:")
    print(
        "Eventatt: {}, Duratt: {}, Relatt: {}, Dropout: {}, Activation: {}, Binomial: {}, concat_fine2dur: {}, concat_dur2fine:{}, fine_to_dur: {}, dur_to_fine: {} \n".format(
            eventatt,
            duratt,
            relatt,
            drop,
            activ,
            bino_bool,
            concat_fine_to_dur,
            concat_dur_to_fine,
            fine_2_dur,
            dur_2_fine))
    device = torch.device(cuda_device_str if torch.cuda.is_available() else "cpu")

    best_model = TemporalModel(
        embedding_size=1024,
        duration_distr=bino_bool,
        elmo_class=ElmoEmbedder(options_file, weight_file, cuda_device=cuda_device_num),
        mlp_dropout=drop,
        mlp_activation=activ,
        tune_embed_size=256,
        event_attention=eventatt,
        dur_attention=duratt,
        rel_attention=relatt,
        concat_fine_to_dur=concat_fine_to_dur,
        concat_dur_to_fine=concat_dur_to_fine,
        fine_to_dur=fine_2_dur,
        dur_to_fine=dur_2_fine,
        fine_squash=True,
        baseline=False,
        dur_MLP_sizes=[128], fine_MLP_sizes=[128],
        dur_output_size=11, fine_output_size=4,
        device=device)

    best_model.load_state_dict(torch.load(model_path + file_path, map_location=cuda_device_str))
    best_model.to(device)

    p1_dur_yhat, p2_dur_yhat, fine_yhat, rel_yhat = predict_fine_dur_only(X, best_model)
    print("Relative timelines completed!!\n")
    ## Store predictions in the dataset
    df['pred1_duration'] = p1_dur_yhat.cpu().numpy()
    df['pred2_duration'] = p2_dur_yhat.cpu().numpy()
    df['b1'] = [b1 for b1, d1, b2, d2 in fine_yhat.cpu().numpy()]
    df['d1'] = [d1 for b1, d1, b2, d2 in fine_yhat.cpu().numpy()]
    df['e1'] = df['b1'] + df['d1']
    df['b2'] = [b2 for b1, d1, b2, d2 in fine_yhat.cpu().numpy()]
    df['d2'] = [d2 for b1, d1, b2, d2 in fine_yhat.cpu().numpy()]
    df['e2'] = df['b2'] + df['d2']
    df = df.drop(['d1', 'd2'], axis=1)
    df['sent_pred_id1'] = df['sentence_id_1'] + " " + df['pred1_root_token'].map(lambda x: str(x))
    df['sent_pred_id2'] = df['sentence_id_2'] + " " + df['pred2_root_token'].map(lambda x: str(x))

    ## Document Timelines
    pred_dict, num_preds, local_data = extract_preds(df)

    ## Run Timeline Model on current docid's data
    model = TimelineModel(data=local_data,
                          num_preds=num_preds,
                          device=torch.device(type="cpu"))

    print("###########   Creating document timelines    ###########")
    pred_b1, pred_e1, pred_b2, pred_e2, pred_timeline = model.fit(local_data, epochs=5000)

    preds_arr = local_data[['sent_pred_id1', 'sent_pred_id2']].values
    uniq_preds = np.unique(preds_arr.flatten())
    # print(uniq_preds)

    preds_text = extract_pred_text(uniq_preds, local_data)

    ans_df = pd.DataFrame(data=pred_timeline,
                          columns=['start_pt', 'duration'])
    ans_df['sent_pred_id'] = uniq_preds
    ans_df['pred_text'] = preds_text

    ## Save prediction files
    ans_df.to_csv(args.outpath + "/" + filename + "_timeline.csv", index=False)
    local_data.to_csv(args.outpath + "/" + filename + "_predictions.csv", index=False)

    print("\nOutput written to the predictions folder.")


if __name__ == "__main__":
    main()

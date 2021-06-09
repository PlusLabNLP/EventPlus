import torch
import torch.nn as nn
import argparse
import os
import sys
import apex.amp as amp
from time import time
import pandas as pd
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from os.path import dirname, abspath
parent_dir = dirname(dirname(abspath(__file__)))
sys.path.insert(0, parent_dir)
from utils import str2bool, print_log, setup_logger, compute_eval_metrics, compute_predictions
from preprocess import TempEveDataset
from scripts.utils import TemporalModel, ElmoEmbedder

"""
Train w/ Val:

Test:
python3 main.py \
--mode test \
--model elmo_mlp \
--model_ckpt model_ckpt/model_param_param_param_1_0_128_128_0_0_0_0_0.0_0.5_relu_1.pth \
--inp_file ./mu_dev_out.json \
--pred_file ./predict_out.csv \
--batch 1
"""


def main():
    parser = argparse.ArgumentParser(description='UDS-T Expt')

    # Experiment params
    parser.add_argument('--mode',           type=str,       help='train or test mode', required=True, choices=['train', 'test'])
    parser.add_argument('--expt_dir',       type=str,       help='root directory to save model & summaries')
    parser.add_argument('--expt_name',      type=str,       help='expt_dir/expt_name: organize experiments')
    parser.add_argument('--run_name',       type=str,       help='expt_dir/expt_name/run_name: organize training runs')

    # Model params
    parser.add_argument('--model',          type=str,       help='model type', choices=['elmo_mlp', 'bert'], required=True)
    parser.add_argument('--model_ckpt',     type=str,       help='path to model checkpoint .pth file')
    parser.add_argument('--pretrain',       type=str2bool,  help='use pretrained encoder', default='true')

    # Data params
    parser.add_argument('--inp_file',       type=str,       help='path to dataset file (json/tsv)', required=True)
    parser.add_argument('--pred_file',      type=str,       help='output prediction csv file (input, prediction)')

    # Training params
    parser.add_argument('--lr',             type=float,     help='learning rate', default=1e-5)
    parser.add_argument('--epochs',         type=int,       help='number of epochs', default=50)
    parser.add_argument('--batch_size',     type=int,       help='batch size', default=8)
    parser.add_argument('--log_interval',   type=int,       help='interval size for logging training summaries', default=100)
    parser.add_argument('--save_interval',  type=int,       help='save model after `n` weight update steps', default=10000)
    parser.add_argument('--val_size',       type=int,       help='validation set size for evaluating metrics', default=1024)

    # GPU params
    parser.add_argument('--gpu_id',         type=int,       help='cuda:gpu_id (0,1,2,..) if num_gpus = 1', default=0)
    parser.add_argument('--opt_lvl',        type=int,       help='Automatic-Mixed Precision: opt-level (O_)', default=1, choices=[0, 1, 2, 3])
    # parser.add_argument('--num_gpus',    type=int,   help='number of GPUs to use for training', default=1)

    # Misc params
    parser.add_argument('--num_workers',    type=int,       help='number of worker threads for Dataloader', default=1)

    args = parser.parse_args()

    device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    print('Selected Device: {}'.format(device))

    # Set CUDA device
    torch.cuda.set_device(device)

    # Train params
    n_epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr

    # Train
    if args.mode == 'train':
        # Setup train log directory
        log_dir = os.path.join(args.expt_dir, args.expt_name, args.run_name)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # TensorBoard summaries setup  -->  /expt_dir/expt_name/run_name/
        writer = SummaryWriter(log_dir)

        # Train log file
        log_file = setup_logger(parser, log_dir)

        print('Training Log Directory: {}\n'.format(log_dir))

        # Load & split dataset
        img_train, annot_train, img_val, annot_val, idx2label = VOCContextDataset.load_dataset(args.data_dir, split=.85)

        # Dataset & Dataloader
        train_dataset = VOCContextDataset(img_train, annot_train, img_size, transforms_train)
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)

        val_dataset = VOCContextDataset(img_val, annot_val, img_size, transforms_eval)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)

        # Fold sizes
        train_size = train_dataset.__len__()
        val_size = val_dataset.__len__()
        log_msg = 'Train Data Size: {}\n'.format(train_size)
        log_msg += 'Validation Data Size: {}\n\n'.format(val_size)

        # Min of the total & subset size
        val_used_size = min(val_size, args.val_size)
        log_msg += 'Validation Accuracy is computed using {} samples. See --val_size\n'.format(val_used_size)
        # Plot segmentation maps
        # from utils import plot_segmentation_map
        # batch = next(iter(train_loader))
        # for label_map in batch['label']:
        #     plot_segmentation_map(label_map)
        # sys.exit()
        num_cls = len(idx2label)

        log_msg += 'Total Number of Classes {}\n'.format(num_cls)
        print_log(log_msg, log_file)

        # Build Model
        model = FCN8s(args.encoder, num_cls, args.pretrain)
        model.to(device)

        # Loss & Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr)

        model, optimizer = amp.initialize(model, optimizer, opt_level="O{}".format(args.opt_lvl))

        # Step & Epoch
        start_epoch = 1
        curr_step = 1
        best_val_acc = 0.0

        # Load model checkpoint file (if specified)
        if args.model_ckpt:
            checkpoint = torch.load(args.model_ckpt, map_location=device)

            # Load model & optimizer
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Load other info
            curr_step = checkpoint['curr_step']
            start_epoch = checkpoint['epoch']
            prev_loss = checkpoint['loss']

            log_msg = 'Resuming Training...\n'
            log_msg += 'Model successfully loaded from {}\n'.format(args.model_ckpt)
            log_msg += 'Training loss: {:2f} (from ckpt)\n'.format(prev_loss)

            print_log(log_msg, log_file)

        steps_per_epoch = len(train_loader)
        start_time = time()

        for epoch in range(start_epoch, start_epoch + n_epochs):
            for batch in train_loader:
                # Load to device, for the list of batch tensors
                image = batch['image'].to(device)
                label = batch['label'].to(device)

                # Forward Pass
                label_logits = model(image)

                # Compute Loss
                loss = criterion(label_logits, label)

                # Backward Pass
                optimizer.zero_grad()

                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()

                optimizer.step()

                # Print Results - Loss value & Validation Accuracy
                if curr_step % args.log_interval == 0 or curr_step == 1:
                    # Validation set accuracy
                    if args.val_csv:
                        validation_metrics = compute_eval_metrics(model, val_loader, device, val_used_size)

                        # Reset the mode to training
                        model.train()

                        log_msg = 'Validation Accuracy: {:.2f} %  || Validation Loss: {:.4f}'.format(
                                validation_metrics['accuracy'], validation_metrics['loss'])

                        print_log(log_msg, log_file)

                        # Add summaries to TensorBoard
                        writer.add_scalar('Val/Loss', validation_metrics['loss'], curr_step)
                        writer.add_scalar('Val/Accuracy', validation_metrics['accuracy'], curr_step)

                    # Add summaries to TensorBoard
                    writer.add_scalar('Train/Loss', loss.item(), curr_step)

                    # Compute elapsed & remaining time for training to complete
                    time_elapsed = (time() - start_time) / 3600
                    total_time = (time_elapsed / curr_step) * steps_per_epoch * n_epochs
                    time_left = total_time - time_elapsed

                    log_msg = 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} | time elapsed: {:.2f}h | time left: {:.2f}h'.format(
                            epoch, n_epochs, curr_step, steps_per_epoch, loss.item(), time_elapsed, time_left)

                    print_log(log_msg, log_file)

                # Save the model
                if curr_step % args.save_interval == 0:
                    path = os.path.join(log_dir, 'model_' + str(curr_step) + '.pth')

                    state_dict = {'model_state_dict': model.state_dict(),
                                  'optimizer_state_dict': optimizer.state_dict(),
                                  'curr_step': curr_step, 'loss': loss.item(),
                                  'epoch': epoch, 'label_set': idx2label,
                                  'val_accuracy': best_val_acc}

                    torch.save(state_dict, path)

                    log_msg = 'Saving the model at the {} step to directory:{}'.format(curr_step, log_dir)
                    print_log(log_msg, log_file)

                curr_step += 1

            # Validation set accuracy on the entire set
            if args.val_csv:
                # Total validation set size
                total_validation_size = val_dataset.__len__()
                validation_metrics = compute_eval_metrics(model, val_loader, device, total_validation_size)

                log_msg = '\nAfter {} epoch:\n'.format(epoch)
                log_msg += 'Validation Accuracy: {:.2f} %  || Validation Loss: {:.4f}\n'.format(
                    validation_metrics['accuracy'], validation_metrics['loss'])

                print_log(log_msg, log_file)

                # Update the best validation accuracy
                if validation_metrics["accuracy"] > best_val_acc:
                    best_val_acc = validation_metrics["accuracy"]

                    filename = 'ep_{}_acc_{:.4f}_model.pth'.format(epoch + 1, best_val_acc)
                    path = os.path.join(log_dir, filename)

                    state_dict = {'model_state_dict': model.state_dict(),
                                  'optimizer_state_dict': optimizer.state_dict(),
                                  'curr_step': curr_step, 'loss': loss.item(),
                                  'epoch': epoch, 'label_set': idx2label,
                                  'val_accuracy': best_val_acc}

                    torch.save(state_dict, path)

                    log_msg = "** Best Performing Model: {:.2f} ** \nSaving weights at {}\n".format(best_val_acc, path)
                    print_log(log_msg, log_file)

                # Reset the mode to training
                model.train()

        writer.close()
        log_file.close()

    elif args.mode == 'test':
        # Model Configs
        options_file = "./scripts/elmo_files/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "./scripts/elmo_files/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
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

        print("Eventatt: {}, Duratt: {}, Relatt: {}, Dropout: {}, Activation: {}, Binomial: {}, "
              "concat_fine2dur: {}, concat_dur2fine:{}, fine_to_dur: {}, dur_to_fine: {} \n"
                .format(eventatt, duratt, relatt, drop, activ, bino_bool,
                        concat_fine_to_dur, concat_dur_to_fine, fine_2_dur, dur_2_fine))

        # Dataloader
        test_dataset = TempEveDataset(args.inp_file)

        test_loader = DataLoader(test_dataset, args.batch_size, num_workers=args.num_workers, drop_last=False)

        # Model
        model = TemporalModel(embedding_size=1024, duration_distr=bino_bool,
                              elmo_class=ElmoEmbedder(options_file, weight_file, cuda_device=args.gpu_id),
                              mlp_dropout=drop, mlp_activation=activ, tune_embed_size=256, event_attention=eventatt,
                              dur_attention=duratt, rel_attention=relatt, concat_fine_to_dur=concat_fine_to_dur,
                              concat_dur_to_fine=concat_dur_to_fine, fine_to_dur=fine_2_dur, dur_to_fine=dur_2_fine,
                              fine_squash=True, baseline=False, dur_MLP_sizes=[128], fine_MLP_sizes=[128],
                              dur_output_size=11, fine_output_size=4, device=device)
        model.to(device)

        # Load model weights
        checkpoint = torch.load(args.model_ckpt, map_location=device)
        model.load_state_dict(checkpoint if args.model == 'elmo_mlp' else checkpoint['model_state_dict'])

        # Inference
        outputs = compute_predictions(model, test_loader, device)

        # Save predictions
        idx2label = ['inst', 'secs', 'mins', 'hours', 'days', 'weeks', 'months', 'years', 'decades', 'cents', 'forever']

        # DataFrame
        df_out = pd.DataFrame(outputs)[['p1_dur', 'root', 'sentence']]

        # Map duration index to label
        df_out['p1_dur'] = df_out['p1_dur'].apply(lambda idx: idx2label[idx])

        df_out.to_csv(args.pred_file, index=False)

        print('Total samples: {}\n'.format(df_out.__len__()))
        print('Results saved at {}\n'.format(args.pred_file))


if __name__ == '__main__':
    main()

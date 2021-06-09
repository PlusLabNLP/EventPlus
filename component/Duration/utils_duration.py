import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


idx2label = ['inst', 'secs', 'mins', 'hours', 'days', 'weeks', 'months', 'years', 'decades', 'cents', 'forever']


@torch.no_grad()
def compute_predictions(model, dataloader):
    """
    Computes model outputs.

    :param model: model to evaluate
    :param dataloader: validation/test dataset loader
    :return: outputs
    :rtype: dict
    """
    model.eval()
    outputs = {'sentence': [], 'root_text': [], 'root_idx': [],
               'p1_dur': [], 'p2_dur': [], 'fine': [], 'rel': []}

    # Evaluate on mini-batches & then average over the total
    for batch in dataloader:
        # Load to device, for the list of batch tensors
        words = batch['words_list']         # .to(device)
        root = batch['root_idx']            # .to(device)

        # Add dummy event (2)
        span = [[[x], [x]] for x in root.tolist()]
        root = [[x, x] for x in root.tolist()]

        # Convert words to batch-first: [L, B] --> [B, L]
        words = list(map(list, zip(*words)))

        # Forward Pass
        p1_dur, p2_dur, fine, rel = model(words, span, root)

        _, p1_dur = p1_dur.max(1)
        _, p2_dur = p2_dur.max(1)

        outputs['sentence'] += [' '.join(w_lst) for w_lst in words]
        outputs['root_text'] += batch['root_text']
        outputs['root_idx'] += [idx.item() for idx in batch['root_idx']]

        outputs['p1_dur'] += p1_dur.detach().cpu().tolist()
        outputs['p2_dur'] += p2_dur.detach().cpu().tolist()
        outputs['fine'] += fine.detach().cpu().tolist()
        outputs['rel'] += rel.detach().cpu().tolist()

    return outputs


@torch.no_grad()
def compute_eval_metrics(model, dataloader, device, size):
    """
    For the given model, computes accuracy & loss on validation/test set.

    :param model: model to evaluate
    :param dataloader: validation/test set dataloader
    :param device: cuda/cpu device where the model resides
    :param size: no. of samples (subset) to use
    :return: metrics {'accuracy', 'loss'}
    :rtype: dict
    """
    model.eval()

    loss = 0.0
    num_correct = 0
    total_samples = 0

    # Evaluate on mini-batches & then average over the total
    for n_iter, batch in enumerate(dataloader):
        # Load to device, for the list of batch tensors
        image = batch['image'].to(device)
        label = batch['label'].to(device)

        # Forward Pass
        label_logits = model(image)

        # Compute Accuracy
        label_predicted = torch.argmax(label_logits, dim=1)
        correct = (label == label_predicted)
        num_correct += correct.sum().item()

        # Compute Loss
        loss += F.cross_entropy(label_logits, label, reduction='mean')

        batch_size = label_logits.shape[0]
        total_samples += batch_size

        if total_samples > size:
            break

    # Final Accuracy
    accuracy = 100.0 * (num_correct / total_samples)

    # Final Loss (averaged over mini-batches - n_iter)
    loss = loss / n_iter

    metrics = {'accuracy': accuracy, 'loss': loss}

    return metrics


# ---------------------------------------------------------------------------
def setup_logger(parser, log_dir, file_name='train_log.txt'):
    """
    Generates log file and writes the executed python flags for the current run,
    along with the training log (printed to console). \n

    This is helpful in maintaining experiment logs (with arguments). \n

    While resuming training, the new output log is simply appended to the previously created train log file.

    :param parser: argument parser object
    :param log_dir: file path (to create)
    :param file_name: log file name
    :return: train log file
    """
    log_file_path = os.path.join(log_dir, file_name)

    log_file = open(log_file_path, 'a+')

    # python3 file_name.py
    log_file.write('python3 ' + sys.argv[0] + '\n')

    # Add all the arguments (key value)
    args = parser.parse_args()

    for key, value in vars(args).items():
        # write to train log file
        log_file.write('--' + key + ' ' + str(value) + '\n')

    log_file.write('\n\n')
    log_file.flush()

    return log_file


def print_log(msg, log_file):
    """
    :param str msg: Message to be printed & logged
    :param file log_file: log file
    """
    log_file.write(msg + '\n')
    log_file.flush()

    print(msg)


def str2bool(v):
    v = v.lower()
    assert v in ['true', 'false', '1', '0'], 'Option requires: "true" or "false"'
    return v in ['true', '1']

from collections import OrderedDict

def split_tri_output(trigger_seq, B2I):
    '''
    given trigger_sequence, we will generate trigger word indexes for argument module
    Args:
        trigger_seq: a list of int that represent the trigger sequence.
        label_to_id_tri: a map that mapping from BIO labels ro index
    Return:
        A list of lists of trigger word idx. E.g. [[1,2], [5,6], [10]]
    '''
    tri_idx = []
    tri_type = []
    in_chunk = False
    curr_idx = []
    curr_I = None
    for i in range(len(trigger_seq)):
        # end of chunk
        if in_chunk:
            if trigger_seq[i] != curr_I:
                tri_idx.append(curr_idx)
                tri_type.append(curr_I - 1)   # -1 accounts for finding the id of B-xxx
                curr_idx = []
                curr_I = None
                in_chunk = False
            elif trigger_seq[i] == curr_I:
                curr_idx.append(i)
                if i == len(trigger_seq) - 1:
                    # the last token is a I token
                    tri_idx.append(curr_idx)
                    tri_type.append(curr_I - 1)   # -1 accounts for finding the id of B-xxx

        # beginning of chunk
        if trigger_seq[i] in B2I:
            curr_idx = [i]
            in_chunk = True
            curr_I = B2I[trigger_seq[i]]
            if i == len(trigger_seq) - 1:
                # the last token is a B token
                tri_idx.append(curr_idx)
                tri_type.append(curr_I - 1)   # -1 accounts for finding the id of B-xxx

    assert len(tri_idx) == len(tri_type)
    return tri_idx, tri_type

if __name__ == '__main__':
    label_to_id_t = OrderedDict([('O', 1), ('B-ANCHOR', 2), ('I-ANCHOR', 3), ('<PAD>', 0)])
    fake_data = ['O', 'B-ANCHOR', 'I-ANCHOR', 'I-ANCHOR', 'B-ANCHOR', 'O', 'O', 'B-ANCHOR', 'I-ANCHOR', 'O', 'B-ANCHOR', 'I-ANCHOR']
    print(fake_data)
    fake_data = [label_to_id_t[x] for x in fake_data]
    tri_idx = split_tri_output(fake_data, label_to_id_t)
    print(tri_idx)

import numpy as np
import torch

from torch.nn.utils.rnn import pad_sequence

if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor

def match_seq_len_p3(q_seqs, r_seqs, a_seqs, b_seqs, c_seqs, seq_len, pad_val=-1):
    '''
        Args:
            q_seqs: the question(KC) sequences with the size of \
                [batch_size, some_sequence_length]
            r_seqs: the response sequences with the size of \
                [batch_size, some_sequence_length]

            Note that the "some_sequence_length" is not uniform over \
                the whole batch of q_seqs and r_seqs

            seq_len: the sequence length to match the q_seqs, r_seqs \
                to same length
            pad_val: the padding value for the sequence with the length \
                longer than seq_len

        Returns:
            proc_q_seqs: the processed q_seqs with the size of \
                [batch_size, seq_len + 1]
            proc_r_seqs: the processed r_seqs with the size of \
                [batch_size, seq_len + 1]
    '''
    proc_q_seqs = []
    proc_r_seqs = []
    proc_a_seqs = []
    proc_b_seqs = []
    proc_c_seqs = []

    for q_seq, r_seq, a_seq, b_seq, c_seq in zip(q_seqs, r_seqs, a_seqs, b_seqs, c_seqs):
        i = 0
        while i + seq_len + 1 < len(q_seq):
            proc_q_seqs.append(q_seq[i:i + seq_len + 1])
            proc_r_seqs.append(r_seq[i:i + seq_len + 1])
            proc_a_seqs.append(a_seq[i:i + seq_len + 1])
            proc_b_seqs.append(b_seq[i:i + seq_len + 1])
            proc_c_seqs.append(c_seq[i:i + seq_len + 1])
            i += seq_len + 1

        proc_q_seqs.append(
            np.concatenate(
                [
                    q_seq[i:],
                    np.array([pad_val] * (i + seq_len + 1 - len(q_seq)))
                ]
            )
        )
        proc_r_seqs.append(
            np.concatenate(
                [
                    r_seq[i:],
                    np.array([pad_val] * (i + seq_len + 1 - len(q_seq)))
                ]
            )
        )

        proc_a_seqs.append(
            np.concatenate(
                [
                    a_seq[i:],
                    np.array([pad_val] * (i + seq_len + 1 - len(q_seq)))
                ]
            )
        )

        proc_b_seqs.append(
            np.concatenate(
                [
                    b_seq[i:],
                    np.array([pad_val] * (i + seq_len + 1 - len(q_seq)))
                ]
            )
        )

        proc_c_seqs.append(
            np.concatenate(
                [
                    c_seq[i:],
                    np.array([pad_val] * (i + seq_len + 1 - len(q_seq)))
                ]
            )
        )
    return proc_q_seqs, proc_r_seqs, proc_a_seqs, proc_b_seqs, proc_c_seqs

def match_seq_len_p2(q_seqs, r_seqs, a_seqs, b_seqs, seq_len, pad_val=-1):
    '''
        Args:
            q_seqs: the question(KC) sequences with the size of \
                [batch_size, some_sequence_length]
            r_seqs: the response sequences with the size of \
                [batch_size, some_sequence_length]

            Note that the "some_sequence_length" is not uniform over \
                the whole batch of q_seqs and r_seqs

            seq_len: the sequence length to match the q_seqs, r_seqs \
                to same length
            pad_val: the padding value for the sequence with the length \
                longer than seq_len

        Returns:
            proc_q_seqs: the processed q_seqs with the size of \
                [batch_size, seq_len + 1]
            proc_r_seqs: the processed r_seqs with the size of \
                [batch_size, seq_len + 1]
    '''
    proc_q_seqs = []
    proc_r_seqs = []
    proc_a_seqs = []
    proc_b_seqs = []
    for q_seq, r_seq, a_seq, b_seq in zip(q_seqs, r_seqs, a_seqs, b_seqs):
        i = 0
        while i + seq_len + 1 < len(q_seq):
            proc_q_seqs.append(q_seq[i:i + seq_len + 1])
            proc_r_seqs.append(r_seq[i:i + seq_len + 1])
            proc_a_seqs.append(a_seq[i:i + seq_len + 1])
            proc_b_seqs.append(b_seq[i:i + seq_len + 1])
            i += seq_len + 1

        proc_q_seqs.append(
            np.concatenate(
                [
                    q_seq[i:],
                    np.array([pad_val] * (i + seq_len + 1 - len(q_seq)))
                ]
            )
        )
        proc_r_seqs.append(
            np.concatenate(
                [
                    r_seq[i:],
                    np.array([pad_val] * (i + seq_len + 1 - len(q_seq)))
                ]
            )
        )

        proc_a_seqs.append(
            np.concatenate(
                [
                    a_seq[i:],
                    np.array([pad_val] * (i + seq_len + 1 - len(q_seq)))
                ]
            )
        )

        proc_b_seqs.append(
            np.concatenate(
                [
                    b_seq[i:],
                    np.array([pad_val] * (i + seq_len + 1 - len(q_seq)))
                ]
            )
        )

    return proc_q_seqs, proc_r_seqs, proc_a_seqs, proc_b_seqs

def match_seq_len_p1(q_seqs, r_seqs, a_seqs, seq_len, pad_val=-1):
    '''
        Args:
            q_seqs: the question(KC) sequences with the size of \
                [batch_size, some_sequence_length]
            r_seqs: the response sequences with the size of \
                [batch_size, some_sequence_length]

            Note that the "some_sequence_length" is not uniform over \
                the whole batch of q_seqs and r_seqs

            seq_len: the sequence length to match the q_seqs, r_seqs \
                to same length
            pad_val: the padding value for the sequence with the length \
                longer than seq_len

        Returns:
            proc_q_seqs: the processed q_seqs with the size of \
                [batch_size, seq_len + 1]
            proc_r_seqs: the processed r_seqs with the size of \
                [batch_size, seq_len + 1]
    '''
    proc_q_seqs = []
    proc_r_seqs = []
    proc_a_seqs = []

    for q_seq, r_seq, a_seq in zip(q_seqs, r_seqs, a_seqs):
        i = 0
        while i + seq_len + 1 < len(q_seq):
            proc_q_seqs.append(q_seq[i:i + seq_len + 1])
            proc_r_seqs.append(r_seq[i:i + seq_len + 1])
            proc_a_seqs.append(a_seq[i:i + seq_len + 1])
            i += seq_len + 1

        proc_q_seqs.append(
            np.concatenate(
                [
                    q_seq[i:],
                    np.array([pad_val] * (i + seq_len + 1 - len(q_seq)))
                ]
            )
        )
        proc_r_seqs.append(
            np.concatenate(
                [
                    r_seq[i:],
                    np.array([pad_val] * (i + seq_len + 1 - len(q_seq)))
                ]
            )
        )

        proc_a_seqs.append(
            np.concatenate(
                [
                    a_seq[i:],
                    np.array([pad_val] * (i + seq_len + 1 - len(q_seq)))
                ]
            )
        )

    return proc_q_seqs, proc_r_seqs, proc_a_seqs

def match_seq_len(q_seqs, r_seqs, seq_len, pad_val=-1):
    '''
        Args:
            q_seqs: the question(KC) sequences with the size of \
                [batch_size, some_sequence_length]
            r_seqs: the response sequences with the size of \
                [batch_size, some_sequence_length]

            Note that the "some_sequence_length" is not uniform over \
                the whole batch of q_seqs and r_seqs

            seq_len: the sequence length to match the q_seqs, r_seqs \
                to same length
            pad_val: the padding value for the sequence with the length \
                longer than seq_len

        Returns:
            proc_q_seqs: the processed q_seqs with the size of \
                [batch_size, seq_len + 1]
            proc_r_seqs: the processed r_seqs with the size of \
                [batch_size, seq_len + 1]
    '''
    proc_q_seqs = []
    proc_r_seqs = []

    for q_seq, r_seq in zip(q_seqs, r_seqs):
        i = 0
        while i + seq_len + 1 < len(q_seq):
            proc_q_seqs.append(q_seq[i:i + seq_len + 1])
            proc_r_seqs.append(r_seq[i:i + seq_len + 1])

            i += seq_len + 1

        proc_q_seqs.append(
            np.concatenate(
                [
                    q_seq[i:],
                    np.array([pad_val] * (i + seq_len + 1 - len(q_seq)))
                ]
            )
        )
        proc_r_seqs.append(
            np.concatenate(
                [
                    r_seq[i:],
                    np.array([pad_val] * (i + seq_len + 1 - len(q_seq)))
                ]
            )
        )

    return proc_q_seqs, proc_r_seqs


def collate_fn(batch, pad_val=-1):
    '''
        The collate function for torch.utils.data.DataLoader

        Returns:
            q_seqs: the question(KC) sequences with the size of \
                [batch_size, maximum_sequence_length_in_the_batch]
            r_seqs: the response sequences with the size of \
                [batch_size, maximum_sequence_length_in_the_batch]
            qshft_seqs: the question(KC) sequences which were shifted \
                one step to the right with ths size of \
                [batch_size, maximum_sequence_length_in_the_batch]
            rshft_seqs: the response sequences which were shifted \
                one step to the right with ths size of \
                [batch_size, maximum_sequence_length_in_the_batch]
            mask_seqs: the mask sequences indicating where \
                the padded entry is with the size of \
                [batch_size, maximum_sequence_length_in_the_batch]
    '''
    q_seqs = []
    r_seqs = []
    qshft_seqs = []
    rshft_seqs = []

    for q_seq, r_seq in batch:
        q_seqs.append(FloatTensor(q_seq[:-1]))
        r_seqs.append(FloatTensor(r_seq[:-1]))
        qshft_seqs.append(FloatTensor(q_seq[1:]))
        rshft_seqs.append(FloatTensor(r_seq[1:]))

    q_seqs = pad_sequence(
        q_seqs, batch_first=True, padding_value=pad_val
    )
    r_seqs = pad_sequence(
        r_seqs, batch_first=True, padding_value=pad_val
    )
    qshft_seqs = pad_sequence(
        qshft_seqs, batch_first=True, padding_value=pad_val
    )
    rshft_seqs = pad_sequence(
        rshft_seqs, batch_first=True, padding_value=pad_val
    )

    mask_seqs = (q_seqs != pad_val) * (qshft_seqs != pad_val)

    q_seqs, r_seqs, qshft_seqs, rshft_seqs = \
        q_seqs * mask_seqs, r_seqs * mask_seqs, qshft_seqs * mask_seqs, \
        rshft_seqs * mask_seqs

    return q_seqs, r_seqs, qshft_seqs, rshft_seqs, mask_seqs


def collate_fn_p1(batch, pad_val=-1):
    '''
        The collate function for torch.utils.data.DataLoader

        Returns:
            q_seqs: the question(KC) sequences with the size of \
                [batch_size, maximum_sequence_length_in_the_batch]
            r_seqs: the response sequences with the size of \
                [batch_size, maximum_sequence_length_in_the_batch]
            qshft_seqs: the question(KC) sequences which were shifted \
                one step to the right with ths size of \
                [batch_size, maximum_sequence_length_in_the_batch]
            rshft_seqs: the response sequences which were shifted \
                one step to the right with ths size of \
                [batch_size, maximum_sequence_length_in_the_batch]
            mask_seqs: the mask sequences indicating where \
                the padded entry is with the size of \
                [batch_size, maximum_sequence_length_in_the_batch]
    '''
    q_seqs = []
    r_seqs = []
    qshft_seqs = []
    rshft_seqs = []
    a_seqs = []
    ashft_seqs = []


    for q_seq, r_seq, a_seq in batch:
        q_seqs.append(FloatTensor(q_seq[:-1]))
        r_seqs.append(FloatTensor(r_seq[:-1]))
        qshft_seqs.append(FloatTensor(q_seq[1:]))
        rshft_seqs.append(FloatTensor(r_seq[1:]))
        a_seqs.append(FloatTensor(a_seq[:-1]))
        ashft_seqs.append(FloatTensor(a_seq[1:]))

    q_seqs = pad_sequence(
        q_seqs, batch_first=True, padding_value=pad_val
    )
    r_seqs = pad_sequence(
        r_seqs, batch_first=True, padding_value=pad_val
    )
    qshft_seqs = pad_sequence(
        qshft_seqs, batch_first=True, padding_value=pad_val
    )
    rshft_seqs = pad_sequence(
        rshft_seqs, batch_first=True, padding_value=pad_val
    )

    a_seqs = pad_sequence(
        a_seqs, batch_first=True, padding_value=pad_val
    )

    ashft_seqs = pad_sequence(
        ashft_seqs, batch_first=True, padding_value=pad_val
    )

    # interesting here
    # may use another mask
    mask_seqs = (q_seqs != pad_val) * (qshft_seqs != pad_val)

    q_seqs, r_seqs, qshft_seqs, rshft_seqs, a_seqs, ashft_seqs = \
        q_seqs * mask_seqs, r_seqs * mask_seqs, qshft_seqs * mask_seqs, \
        rshft_seqs * mask_seqs, a_seqs * mask_seqs, ashft_seqs * mask_seqs

    return q_seqs, r_seqs, qshft_seqs, rshft_seqs, a_seqs, ashft_seqs, mask_seqs

def collate_fn_p2(batch, pad_val=-1):
    '''
        The collate function for torch.utils.data.DataLoader

        Returns:
            q_seqs: the question(KC) sequences with the size of \
                [batch_size, maximum_sequence_length_in_the_batch]
            r_seqs: the response sequences with the size of \
                [batch_size, maximum_sequence_length_in_the_batch]
            qshft_seqs: the question(KC) sequences which were shifted \
                one step to the right with ths size of \
                [batch_size, maximum_sequence_length_in_the_batch]
            rshft_seqs: the response sequences which were shifted \
                one step to the right with ths size of \
                [batch_size, maximum_sequence_length_in_the_batch]
            mask_seqs: the mask sequences indicating where \
                the padded entry is with the size of \
                [batch_size, maximum_sequence_length_in_the_batch]
    '''
    q_seqs = []
    r_seqs = []
    qshft_seqs = []
    rshft_seqs = []
    a_seqs = []
    ashft_seqs = []
    b_seqs = []
    bshft_seqs = []

    for q_seq, r_seq, a_seq, b_seq in batch:
        q_seqs.append(FloatTensor(q_seq[:-1]))
        r_seqs.append(FloatTensor(r_seq[:-1]))
        qshft_seqs.append(FloatTensor(q_seq[1:]))
        rshft_seqs.append(FloatTensor(r_seq[1:]))
        a_seqs.append(FloatTensor(a_seq[:-1]))
        ashft_seqs.append(FloatTensor(a_seq[1:]))

        b_seqs.append(FloatTensor(b_seq[:-1]))
        bshft_seqs.append(FloatTensor(b_seq[1:]))

    q_seqs = pad_sequence(
        q_seqs, batch_first=True, padding_value=pad_val
    )
    r_seqs = pad_sequence(
        r_seqs, batch_first=True, padding_value=pad_val
    )
    qshft_seqs = pad_sequence(
        qshft_seqs, batch_first=True, padding_value=pad_val
    )
    rshft_seqs = pad_sequence(
        rshft_seqs, batch_first=True, padding_value=pad_val
    )

    a_seqs = pad_sequence(
        a_seqs, batch_first=True, padding_value=pad_val
    )

    ashft_seqs = pad_sequence(
        ashft_seqs, batch_first=True, padding_value=pad_val
    )

    b_seqs = pad_sequence(
        b_seqs, batch_first=True, padding_value=pad_val
    )

    bshft_seqs = pad_sequence(
        bshft_seqs, batch_first=True, padding_value=pad_val
    )

    # interesting here
    # may use another mask
    mask_seqs = (q_seqs != pad_val) * (qshft_seqs != pad_val)

    q_seqs, r_seqs, qshft_seqs, rshft_seqs, a_seqs, ashft_seqs, b_seqs, bshft_seqs = \
        q_seqs * mask_seqs, r_seqs * mask_seqs, qshft_seqs * mask_seqs, \
        rshft_seqs * mask_seqs, a_seqs * mask_seqs, ashft_seqs * mask_seqs,  b_seqs * mask_seqs, bshft_seqs * mask_seqs

    return q_seqs, r_seqs, qshft_seqs, rshft_seqs, a_seqs, ashft_seqs, b_seqs, bshft_seqs, mask_seqs



def collate_fn_p3(batch, pad_val=-1):
    '''
        The collate function for torch.utils.data.DataLoader

        Returns:
            q_seqs: the question(KC) sequences with the size of \
                [batch_size, maximum_sequence_length_in_the_batch]
            r_seqs: the response sequences with the size of \
                [batch_size, maximum_sequence_length_in_the_batch]
            qshft_seqs: the question(KC) sequences which were shifted \
                one step to the right with ths size of \
                [batch_size, maximum_sequence_length_in_the_batch]
            rshft_seqs: the response sequences which were shifted \
                one step to the right with ths size of \
                [batch_size, maximum_sequence_length_in_the_batch]
            mask_seqs: the mask sequences indicating where \
                the padded entry is with the size of \
                [batch_size, maximum_sequence_length_in_the_batch]
    '''
    q_seqs = []
    r_seqs = []
    qshft_seqs = []
    rshft_seqs = []
    a_seqs = []
    ashft_seqs = []
    b_seqs = []
    bshft_seqs = []

    c_seqs = []
    cshft_seqs = []
    for q_seq, r_seq, a_seq, b_seq, c_seq in batch:
        q_seqs.append(FloatTensor(q_seq[:-1]))
        r_seqs.append(FloatTensor(r_seq[:-1]))
        qshft_seqs.append(FloatTensor(q_seq[1:]))
        rshft_seqs.append(FloatTensor(r_seq[1:]))
        a_seqs.append(FloatTensor(a_seq[:-1]))
        ashft_seqs.append(FloatTensor(a_seq[1:]))

        b_seqs.append(FloatTensor(b_seq[:-1]))
        bshft_seqs.append(FloatTensor(b_seq[1:]))
        c_seqs.append(FloatTensor(c_seq[:-1]))
        cshft_seqs.append(FloatTensor(c_seq[1:]))

    q_seqs = pad_sequence(
        q_seqs, batch_first=True, padding_value=pad_val
    )
    r_seqs = pad_sequence(
        r_seqs, batch_first=True, padding_value=pad_val
    )
    qshft_seqs = pad_sequence(
        qshft_seqs, batch_first=True, padding_value=pad_val
    )
    rshft_seqs = pad_sequence(
        rshft_seqs, batch_first=True, padding_value=pad_val
    )

    a_seqs = pad_sequence(
        a_seqs, batch_first=True, padding_value=pad_val
    )

    ashft_seqs = pad_sequence(
        ashft_seqs, batch_first=True, padding_value=pad_val
    )

    b_seqs = pad_sequence(
        b_seqs, batch_first=True, padding_value=pad_val
    )

    bshft_seqs = pad_sequence(
        bshft_seqs, batch_first=True, padding_value=pad_val
    )

    c_seqs = pad_sequence(
        c_seqs, batch_first=True, padding_value=pad_val
    )

    cshft_seqs = pad_sequence(
        cshft_seqs, batch_first=True, padding_value=pad_val
    )

    # interesting here
    # may use another mask
    mask_seqs = (q_seqs != pad_val) * (qshft_seqs != pad_val)

    q_seqs, r_seqs, qshft_seqs, rshft_seqs, a_seqs, ashft_seqs, b_seqs, bshft_seqs, c_seqs, cshft_seqs = \
        q_seqs * mask_seqs, r_seqs * mask_seqs, qshft_seqs * mask_seqs, \
        rshft_seqs * mask_seqs, a_seqs * mask_seqs, ashft_seqs * mask_seqs,  b_seqs * mask_seqs, bshft_seqs * mask_seqs,  c_seqs * mask_seqs, cshft_seqs * mask_seqs

    return q_seqs, r_seqs, qshft_seqs, rshft_seqs, a_seqs, ashft_seqs, b_seqs, bshft_seqs,  c_seqs, cshft_seqs, mask_seqs


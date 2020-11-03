
class DefaultConfig(object):

    acid_one_hot = [0 for i in range(20)]
    acid_idex = {j:i for i,j in enumerate("ACDEFGHIKLMNPQRSTVWY")} #ACDEFGHIKLMNPQRSTVWYX

    max_sequence_length = 500

    batch_size = 32*18

    dropout =0.2

    num_workers = 1#5

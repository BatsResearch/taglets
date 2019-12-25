import numpy as np


class TagletExecuter:
    """
    TagletExecuter class (Equivalent to LFApplier)
    """
    def __init__(self, taglets):
        self.taglets = taglets

    def execute(self, images, batch_size=64, use_gpu=True):
        """
        Top: I implement this function in the most straightforward way. If there is a room for optimization,
             please feel free to optimize. Also, I add use_gpu as one of the arguments of this function to
             tell the taglets if they should execute on gpu, but I am still not sure if this is the correct way
             to do it.
        Perform execute function of all Taglets
        :param images: images
        :param batch_size: batch size
        :param use_gpu: a boolean indicating if the taglets should execute on gpu
        :return: A label matrix of size (num_images, num_taglets)
        """
        num_images = images.shape[0]
        num_taglets = len(self.taglets)
        label_matrix = np.zeros((num_images, num_taglets))
        
        ct = 0
        while ct < num_images:
            batch_images = images[ct:min(ct + batch_size, num_images)]
            for i in range(num_taglets):
                label_matrix[ct:min(ct + batch_size, num_images), i] = self.taglets[i].execute(batch_images, use_gpu)
            ct += batch_size
        return label_matrix

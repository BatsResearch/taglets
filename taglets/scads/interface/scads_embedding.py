import gzip
import numpy as np
import pandas as pd
import os

from .scads import Scads

class ScadsEmbedding:
    """
    A class providing connection to Structured Collections of Annotated Data Sets (SCADS).
    """
    frame = None
    small_frame = None
    k = None
    small_k = None

    @staticmethod
    def load(path_to_embeddings):
        """
        Load all embeddings into main memory
        
        :param path_to_embeddings: path to the file containing the embeddings
        :return:
        """
        if ScadsEmbedding.frame is None:
            label_list = []
            with gzip.open(path_to_embeddings, 'rt', encoding='utf-8') as infile:
                nrows_str, ncols_str = infile.readline().rstrip().split()
    
                nrows = int(nrows_str)
                ncols = int(ncols_str)
                arr = np.zeros((nrows, ncols), dtype='f')
                for line in infile:
                    if len(label_list) >= nrows:
                        break
                    items = line.rstrip().split(' ')
                    label = items[0]
                    if label != '</s>':
                        values = [float(x) for x in items[1:]]
                        arr[len(label_list)] = values
                        label_list.append(label)
            df = pd.DataFrame(arr, index=label_list, dtype='f')
            
            if not df.index[1].startswith('/c/'):
                df.index = ['/c/en/' + label for label in df.index]

            if not df.index.is_monotonic_increasing:
                df = df.sort_index()
            
            ScadsEmbedding.k = df.shape[1]
            ScadsEmbedding.small_k = 100
            ScadsEmbedding.frame = df
            ScadsEmbedding.small_frame = df.iloc[:, : ScadsEmbedding.small_k].copy()
            
    @staticmethod
    def get_vector(node):
        """
        Get the embedding of the given ScadsNode
        
        :param node: Target ScadsNode to get its related nodes
        :return: A normalized embedding as a numpy array or None
        """
        if ScadsEmbedding.frame is None:
            raise RuntimeError("Embeddings are not loaded")
        if not node.get_conceptnet_id() in ScadsEmbedding.frame.index:
            return None
        vec = ScadsEmbedding.frame.loc[node.get_conceptnet_id()].values
        normalized_vec = vec / np.linalg.norm(vec)
        return normalized_vec
    
    @staticmethod
    def get_related_nodes(node, limit=20):
        """
        Get the related nodes based on the cosine similarity of their embeddings

        :param node: target ScadsNode to get its related nodes
        :return:
        """
        vec = ScadsEmbedding.get_vector(node)
        small_vec = vec[: ScadsEmbedding.small_k]
        search_frame = ScadsEmbedding.small_frame
        similar_sloppy = ScadsEmbedding._similar_to_vec(search_frame, small_vec, limit=limit * 50)
        similar_choices = ScadsEmbedding._l2_normalize_rows(
            ScadsEmbedding.frame.loc[similar_sloppy.index].astype('f')
        )

        similar = ScadsEmbedding._similar_to_vec(similar_choices, vec, limit=limit)
        similar_concepts = similar.index.values
        print(similar)
        related_nodes = []
        for concept in similar_concepts:
            try:
                related_node = Scads.get_node_by_conceptnet_id(concept)
                related_nodes.append(related_node)
            except:
                print(f'Concept {concept} not found in Scads')
        return related_nodes

    @staticmethod
    def _similar_to_vec(frame, vec, limit=50):
        # - frame and vec should be normalized
        # - frame should not be made of 8-bit ints
        if vec.dot(vec) == 0.:
            return pd.Series(data=[], index=[], dtype='f')
        similarity = frame.dot(vec)
        return similarity.dropna().nlargest(limit)

    @staticmethod
    def _l2_normalize_rows(frame):
        """
            L_2-normalize the rows of this DataFrame, so their lengths in Euclidean
            distance are all 1. This enables cosine similarities to be computed as
            dot-products between these rows.
            Rows of zeroes will be normalized to zeroes, and frames with no rows will
            be returned as-is.
            """
        if frame.shape[0] == 0:
            return frame
        index = frame.index
        return pd.DataFrame(
            data=frame.apply(lambda row: row / np.linalg.norm(row), axis=1), index=index
        )

if __name__ == '__main__':
    dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import time

    st = time.time()
    print('Start loading database')
    Scads.open('/data/datasets/scads.sqlite3')
    print(f'End loading database: {(time.time() - st) / 60} mins')
    
    st = time.time()
    print('Start loading embedding')
    ScadsEmbedding.load('/data/datasets/numberbatch-en-19.08.txt.gz')
    print(f'End loading embedding: {(time.time()-st) / 60} mins')
    
    node = Scads.get_conceptnet_id('dog')
    related_nodes = ScadsEmbedding.get_related_nodes(node)
    related_nodes = [related_node.get_conceptnet_id() for related_node in related_nodes]
    print(f'Related node to dog: {related_nodes}')

    node = Scads.get_conceptnet_id('pen')
    related_nodes = ScadsEmbedding.get_related_nodes(node)
    related_nodes = [related_node.get_conceptnet_id() for related_node in related_nodes]
    print(f'Related node to pen: {related_nodes}')

    node = Scads.get_conceptnet_id('bear')
    related_nodes = ScadsEmbedding.get_related_nodes(node)
    related_nodes = [related_node.get_conceptnet_id() for related_node in related_nodes]
    print(f'Related node to bear: {related_nodes}')
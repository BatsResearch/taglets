import gzip
import marisa_trie
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
            if path_to_embeddings.endswith('.h5'):
                df = pd.read_hdf(path_to_embeddings, 'mat', encoding='utf-8')
            else:
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
            ScadsEmbedding._trie = marisa_trie.Trie(list(df.index))
    
    @staticmethod
    def get_vector(node, is_node=True):
        """
        Get the embedding of the given ScadsNode
        
        :param node: Target ScadsNode to get its related nodes
        :return: A normalized embedding as a numpy array or None
        """
        if ScadsEmbedding.frame is None:
            raise RuntimeError("Embeddings are not loaded")
        if is_node:
            vec = ScadsEmbedding._expanded_vector(node.get_conceptnet_id())
        else:
            vec = ScadsEmbedding._expanded_vector(node)
        normalized_vec = vec / np.linalg.norm(vec)
        return normalized_vec
    
    @staticmethod
    def get_related_nodes(node, limit=50, is_node=True):
        """
        Get the related nodes based on the cosine similarity of their embeddings

        :param node: target ScadsNode/ConceptNet ID to get its related nodes
        :param limit: number of related nodes to get
        :param is_node: whether we are working with ScadsNode or string
        :return: list of ScadsNode if is_node is True, else, list of ConceptNet IDs
        """
        vec = ScadsEmbedding.get_vector(node, is_node)
        small_vec = vec[: ScadsEmbedding.small_k]
        search_frame = ScadsEmbedding.small_frame
        similar_sloppy = ScadsEmbedding._similar_to_vec(search_frame, small_vec, limit=limit * 50)
        similar_choices = ScadsEmbedding._l2_normalize_rows(
            ScadsEmbedding.frame.loc[similar_sloppy.index].astype('f')
        )

        similar = ScadsEmbedding._similar_to_vec(similar_choices, vec, limit=limit)
        similar_concepts = similar.index.values
        if is_node:
            related_nodes = []
            for concept in similar_concepts:
                try:
                    related_node = Scads.get_node_by_conceptnet_id(concept)
                    related_nodes.append(related_node)
                except:
                    print(f'Concept {concept} not found in Scads')
            return related_nodes
        else:
            return similar_concepts

    @staticmethod
    def get_similarity(node1, node2, is_node=True):
        """
        Get cosine similarity between two classes
 
        :param node1: ScadsNode/ConceptNet ID
        :param node2: ScadsNode/ConceptNet ID
        """
        vec1 = ScadsEmbedding.get_vector(node1, is_node)
        vec2 = ScadsEmbedding.get_vector(node2, is_node)
        return vec1.dot(vec2)

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

    @staticmethod
    def _expanded_vector(term):
        """
        Given a term, make a vector representing information from:
        - The vector for the term
        - The vectors for terms that share a sufficiently-long prefix with
          any terms in this list that are out-of-vocabulary
        """
        return ScadsEmbedding._weighted_average(
            ScadsEmbedding.frame, ScadsEmbedding._expand_terms(term)
        )

    @staticmethod
    def _expand_terms(term):
        """
        Given a term, if it is OOV, find approximations to the terms: terms
        that share a prefix that's as long as possible with the given term.
        This helps increase the recall power of the vector space, because it means
        you can find terms that are too infrequent to have their own vector, getting
        a reasonable guess at the vector they might have.
        """
        expanded = [(term, 1.)]
        if term not in ScadsEmbedding.frame.index:
            prefix_weight = 0.01
            prefix_matches = ScadsEmbedding._match_prefix(term, prefix_weight)
            expanded.extend(prefix_matches)
    
        total_weight = sum(abs(weight) for term, weight in expanded)
        if total_weight == 0:
            return []
        else:
            return [
                (term, weight / total_weight) for (term, weight) in expanded
            ]

    @staticmethod
    def _match_prefix(term, prefix_weight):
        results = []
        while term:
            prefixed = ScadsEmbedding._trie.keys(term)
            if prefixed:
                n_prefixed = len(prefixed)
                for prefixed_term in prefixed:
                    results.append((prefixed_term, prefix_weight / n_prefixed))
                break
            term = term[:-1]
        return results

    @staticmethod
    def _weighted_average(frame, weight_series):
        if isinstance(weight_series, list):
            weight_dict = dict(weight_series)
            weight_series = pd.Series(weight_dict)
        vec = np.zeros(frame.shape[1], dtype='f')
    
        for i, label in enumerate(weight_series.index):
            if label in frame.index:
                val = weight_series[i]
                vec += val * frame.loc[label].values
    
        return pd.Series(data=vec, index=frame.columns, dtype='f')
    

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
    
    node = Scads.get_node_by_conceptnet_id('/c/en/dog')
    related_nodes = ScadsEmbedding.get_related_nodes(node)
    related_nodes = [related_node.get_conceptnet_id() for related_node in related_nodes]
    print(f'Related node to dog: {related_nodes}')

    node = Scads.get_node_by_conceptnet_id('/c/en/pen')
    related_nodes = ScadsEmbedding.get_related_nodes(node)
    related_nodes = [related_node.get_conceptnet_id() for related_node in related_nodes]
    print(f'Related node to pen: {related_nodes}')

    node = Scads.get_node_by_conceptnet_id('/c/en/bear')
    related_nodes = ScadsEmbedding.get_related_nodes(node)
    related_nodes = [related_node.get_conceptnet_id() for related_node in related_nodes]
    print(f'Related node to bear: {related_nodes}')
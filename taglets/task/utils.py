from ..scads import ScadsEmbedding

def labels_to_concept_ids(labels):
    ScadsEmbedding.load('/data/datasets/numberbatch-en-19.08.txt.gz')
    return [ScadsEmbedding.get_related_nodes('/c/en/' + label, limit=1, is_node=False)[0] for label in labels]
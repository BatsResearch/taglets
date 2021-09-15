from ..scads import ScadsEmbedding


def labels_to_concept_ids(labels):
    ScadsEmbedding.load('predefined/embeddings/numberbatch-en19.08.txt.gz')
    concept_ids = []
    for label in labels:
        if label in ['oatghurt', 'soyghurt']:
            concept_ids.append('/c/en/' + label)
        else:
            concept_ids.append(ScadsEmbedding.get_related_nodes('/c/en/' + label, limit=1, is_node=False)[0])
    return concept_ids

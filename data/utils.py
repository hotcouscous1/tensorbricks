from __init__ import *


def make_one_hot(
        num_classes: int,
        category_id: int
) -> Tensor:

    return torch.from_numpy(np.eye(num_classes, dtype='int8')[category_id])



def category_filter(
        original_cat: list or tuple,
        missing_cat: list or tuple
) -> dict:

    """
    Created by hotcouscous1.

    It excludes unused categories, and matches remainders and new categories in dictionary.
    It costs only O(1) for accessing.
    """

    valid_cat = list(filter(lambda c: c not in missing_cat, original_cat))
    cat_table = {c: i for i, c in enumerate(valid_cat)}
    return cat_table


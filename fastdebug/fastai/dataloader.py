# AUTOGENERATED! DO NOT EDIT! File to edit: 01_fastai.dataloader.ipynb (unless otherwise specified).

__all__ = ['collate_error']

# Cell
import inflect
from fastcore.basics import patch
from fastai.data.core import TfmdDL
from fastai.data.load import DataLoader, fa_collate, fa_convert

# Cell
def collate_error(e:Exception, batch):
    """
    Raises an explicit error when the batch could not collate, stating
     what items in the batch are different sizes and their types
    """
    p = inflect.engine()
    err = f'Error when trying to collate the data into batches with fa_collate, '
    err += 'at least two tensors in the batch are not the same size.\n\n'
    # we need to iterate through the entire batch and find a mismatch
    length = len(batch[0])
    for idx in range(length): # for each type in the batch
        for i, item in enumerate(batch):
            if i == 0:
                shape_a = item[idx].shape
                type_a = item[idx].__class__.__name__
            elif item[idx].shape != shape_a:
                shape_b = item[idx].shape
                if shape_a != shape_b:
                    err += f'Mismatch found within the {p.ordinal(idx)} axis of the batch and is of type {type_a}:\n'
                    err += f'The first item has shape: {shape_a}\n'
                    err += f'The {p.number_to_words(p.ordinal(i+1))} item has shape: {shape_b}\n\n'
                    err += f'Please include a transform in `after_item` that ensures all data of type {type_a} is the same size'
                    e.args = [err]
                    raise e

# Cell
@patch
def create_batch(self:DataLoader, b):
    "Collate a list of items into a batch."
    func = (fa_collate,fa_convert)[self.prebatched]
    try:
        return func(b)
    except Exception as e:
        if not self.prebatched:
            collate_error(e, b)
        else: raise e

# Cell
@patch
def new(self:TfmdDL, dataset=None, cls=None, **kwargs):
    res = super(TfmdDL, self).new(dataset, cls, do_setup=False, **kwargs)
    if not hasattr(self, '_n_inp') or not hasattr(self, '_types'):
        try:
            self._one_pass()
            res._n_inp,res._types = self._n_inp,self._types
        except Exception as e:
            print("Could not do one pass in your dataloader, there is something wrong in it")
            raise e
    else: res._n_inp,res._types = self._n_inp,self._types
    return res
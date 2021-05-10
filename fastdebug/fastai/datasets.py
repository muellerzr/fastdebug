# AUTOGENERATED! DO NOT EDIT! File to edit: 04_fastai.datasets.ipynb (unless otherwise specified).

__all__ = ['subset_error']

# Cell
from fastcore.basics import patch, store_attr
from fastcore.foundation import L, mask2idxs
from fastcore.transform import Pipeline
from fastcore.xtras import is_listy

from fastai.imports import pv
from fastai.data.core import TfmdLists

# Cell
@patch
def __init__(self:TfmdLists, items, tfms, use_list=None, do_setup=True, split_idx=None, train_setup=True,
                splits=None, types=None, verbose=False, dl_type=None):
    if items is None or len(items) == 0: raise IndexError('Items passed in either has a length of zero or is None')
    super(TfmdLists, self).__init__(items, use_list=use_list)
    if dl_type is not None: self._dl_type = dl_type
    self.splits = L([slice(None),[]] if splits is None else splits).map(mask2idxs)
    if isinstance(tfms,TfmdLists): tfms = tfms.tfms
    if isinstance(tfms,Pipeline): do_setup=False
    self.tfms = Pipeline(tfms, split_idx=split_idx)
    store_attr('types,split_idx')
    if do_setup:
        pv(f"Setting up {self.tfms}", verbose)
        self.setup(train_setup=train_setup)

# Cell
def subset_error(e:IndexError, i:int) -> IndexError:
    """
    IndexError when attempting to grab a non-existant subset in the dataset at index `i`
    """
    args = e.args[0]
    err = f'Tried to grab subset {i} in the Dataset, but it contains no items.\n\n'
    err += args
    e.args = [err]
    raise e

# Cell
@patch
def subset(self:TfmdLists, i:int):
    "New `TfmdLists` with same tfms that only includes items in `i`th split"
    try: return self._new(self._get(self.splits[i]), split_idx=i)
    except IndexError as e: subset_error(e, i)

# Cell
@patch
def setup(self:TfmdLists, train_setup=True):
    "Transform setup with self"
    self.tfms.setup(self, train_setup)
    if len(self) != 0:
        x = super(TfmdLists, self).__getitem__(0) if self.splits is None else super(TfmdLists, self).__getitem__(self.splits[0])[0]
        self.types = []
        for f in self.tfms.fs:
            self.types.append(getattr(f, 'input_types', type(x)))
            x = f(x)
        self.types.append(type(x))
    t = getattr(self, 'types', [])
    if t is None or len(t) == 0: raise Exception("The stored dataset contains no items and `self.types` has not been setup yet")
    types = L(t if is_listy(t) else [t] for t in self.types).concat().unique()
    self.pretty_types = '\n'.join([f'  - {t}' for t in types])
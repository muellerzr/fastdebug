
# fastdebug
> A helpful library for improving torch and fastai errors


## Install

`pip install fastdebug`

## How to use

`fastdebug` is designed around improving the quality of life when dealing with Pytorch and fastai errors, while also including some new sanity checks (fastai only)

### Pytorch

Pytorch now has:
* `device_error`
* `layer_error`

Both can be imported with:
```python
from fastdebug.error.torch import device_error, layer_error
```

`device_error` prints out a much more readable error for when two tensors aren't on the same device:

```python
inp = torch.rand().cuda()
model = model.cpu()
try:
    _ = model(inp)
except Exception as e:
    device_error(e, 'Input type', 'Model weights')
```
And our new log:
```bash
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
<ipython-input-28-981e0ace9c38> in <module>()
      2     model(x)
      3 except Exception as e:
----> 4     device_error(e, 'Input type', 'Model weights')

10 frames
/usr/local/lib/python3.7/dist-packages/torch/tensor.py in __torch_function__(cls, func, types, args, kwargs)
    993 
    994         with _C.DisableTorchFunction():
--> 995             ret = func(*args, **kwargs)
    996             return _convert(ret, cls)
    997 

RuntimeError: Mismatch between weight types

Input type has type: 		 (torch.cuda.FloatTensor)
Model weights have type: 	 (torch.FloatTensor)

Both should be the same.
```

And with `layer_error`, if there is a shape mismatch it will attempt to find the right layer it was at:
```python
inp = torch.rand(5,2, 3)
try:
    m(inp)
except Exception as e:
    layer_error(e, m)
```

```python
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
<ipython-input-84-d4ab91131841> in <module>()
      3     m(inp)
      4 except Exception as e:
----> 5     layer_error(e, m)

<ipython-input-83-ca2dc02cfff4> in layer_error(e, model)
      8     i, layer = get_layer_by_shape(model, shape)
      9     e.args = [f'Size mismatch between input tensors and what the model expects\n\n{args}\n\tat layer {i}: {layer}']
---> 10     raise e

<ipython-input-84-d4ab91131841> in <module>()
      1 inp = torch.rand(5,2, 3)
      2 try:
----> 3     m(inp)
      4 except Exception as e:
      5     layer_error(e, m)

/mnt/d/lib/python3.7/site-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
    725             result = self._slow_forward(*input, **kwargs)
    726         else:
--> 727             result = self.forward(*input, **kwargs)
    728         for hook in itertools.chain(
    729                 _global_forward_hooks.values(),

/mnt/d/lib/python3.7/site-packages/torch/nn/modules/container.py in forward(self, input)
    115     def forward(self, input):
    116         for module in self:
--> 117             input = module(input)
    118         return input
    119 

/mnt/d/lib/python3.7/site-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
    725             result = self._slow_forward(*input, **kwargs)
    726         else:
--> 727             result = self.forward(*input, **kwargs)
    728         for hook in itertools.chain(
    729                 _global_forward_hooks.values(),

/mnt/d/lib/python3.7/site-packages/torch/nn/modules/conv.py in forward(self, input)
    421 
    422     def forward(self, input: Tensor) -> Tensor:
--> 423         return self._conv_forward(input, self.weight)
    424 
    425 class Conv3d(_ConvNd):

/mnt/d/lib/python3.7/site-packages/torch/nn/modules/conv.py in _conv_forward(self, input, weight)
    418                             _pair(0), self.dilation, self.groups)
    419         return F.conv2d(input, weight, self.bias, self.stride,
--> 420                         self.padding, self.dilation, self.groups)
    421 
    422     def forward(self, input: Tensor) -> Tensor:

RuntimeError: Size mismatch between input tensors and what the model expects

Model expected 4-dimensional input for 4-dimensional weight [3, 3, 1, 1], but got 3-dimensional input of size [5, 2, 3] instead
	at layer 1: Conv2d(3, 3, kernel_size=(1, 1), stride=(1, 1))
```

### fastai

Along with the additions above (and are used during `fit`), fastai now has a `Learner.sanity_check` function, which allows you to quickly perform a basic check to ensure that your call to `fit` won't raise any exceptions. They are performed on the CPU for a partial epoch to make sure that `CUDA` device-assist errors can be preemptively found.

To use it simply do:
```python
from fastdebug.fastai import *
from fastai.vision.all import *

learn = Learner(...)
learn.sanity_check()
```

This is also now an argument in `Learner`, set to `False` by default, so that after making your `Learner` a quick check is ensured.

```python
learn = Learner(..., sanity_check=True)
```

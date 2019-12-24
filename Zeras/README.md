# Zeras

A simple tensorflow wrapper


## Description

```
from Zeras.data_parallelism import DataParallelism, get_files_with_ext
from Zeras.data_batcher import DataBatcher

from Zeras.model_settings_baseboard import ModelSettingsBaseboard
from Zeras.model_baseboard import ModelBaseboard, initialize_with_pretrained_ckpt

from Zeras.optim import linear_warmup_and_exp_decayed_lr
from Zeras.optim import linear_warmup_and_polynomial_decayed_lr
from Zeras.optim import adam_optimizer
from Zeras.optim import adam_wd_optimizer
from Zeras.optim import AdamWeightDecayOptimizer

import Zeras.zoo_nn as zrs_nn
import Zeras.zoo_layers as zrs_layers

from Zeras.vocab import Vocab

```


## Installation

From PyPI distribution system:

```
pip install Zeras
```


Or from the code:

```
git clone https://github.com/Li-Ming-Fan/Zeras
cd Zeras
make clean install
```


Or just incorporate the directory Zeras/ to your project.



## Usage

For usage of this package, please refer to the repositories:

https://github.com/Li-Ming-Fan/text_classification

https://github.com/Li-Ming-Fan/transformer-tensorflow

https://github.com/Li-Ming-Fan/pointer-generator-refactored


</br>

## Acknowledgment

The name "Zeras" is a variant from Keras. We use "Z" because it makes the direcory Zeras/ always display in the last (or first) and do not intersect other directories.



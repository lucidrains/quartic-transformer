## Quartic Transformer (wip)

Exploring an idea where one forgets about efficiency and carries out attention on each edge of the nodes (tokens). You can think of it as doing attention on the attention matrix, taking the perspective of the attention matrix as all the directed edges of a fully connected graph.

The hypothesis is that there is a task out there that the (sub)quartic transformer can do that quadratic transformers cannot.

Will also contain a modified implementation of <a href="https://arxiv.org/abs/2107.10342">multistream transformer</a> (which is not quartic, but number of streams times the quadratic).

## Appreciation

- <a href="https://a16z.com/supporting-the-open-source-ai-community/">A16Z Open Source AI Grant Program</a> and <a href="https://huggingface.co/">🤗 Huggingface</a> for the generous sponsorships, as well as my other sponsors, for affording me the independence to open source current artificial intelligence research

## Install

```bash
$ pip install quartic-transformer
```

## Usage

```python
import torch
from quartic_transformer import QuarticTransformer

model = QuarticTransformer(
    num_tokens = 256,
    depth = 2,
    dim = 512,
    dim_edges = 32
)

tokens = torch.randint(0, 256, (1, 128))

logits = model(tokens) # (1, 128, 256)
```

## Todo

- [x] first add a weak taylor linear attention on top of all edges

- [ ] use coordinate descent routing from the node attention matrix to select a subset of edges to update (and do full attention across)

- [x] build multi-stream transformer, but allow exchange of information at the attention matrix, either through residual attention or a small edge-wise feedforward

## Citation

```bibtex
@inproceedings{Keles2022OnTC,
    title   = {On The Computational Complexity of Self-Attention},
    author  = {Feyza Duman Keles and Pruthuvi Maheshakya Wijewardena and Chinmay Hegde},
    booktitle = {International Conference on Algorithmic Learning Theory},
    year    = {2022},
    url     = {https://api.semanticscholar.org/CorpusID:252198880}
}
```

```bibtex
@article{Burtsev2021MultiStreamT,
    title   = {Multi-Stream Transformers},
    author  = {Mikhail S. Burtsev and Anna Rumshisky},
    journal = {ArXiv},
    year    = {2021},
    volume  = {abs/2107.10342},
    url     = {https://api.semanticscholar.org/CorpusID:236171087}
}
```

```bibtex
@misc{Sutton,
    title  = {The Bitter Lesson},
    url    = {http://www.incompleteideas.net/IncIdeas/BitterLesson.html},
    author = {Sutton, Rich}
}
```

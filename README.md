## Quartic Transformer (wip)

Exploring an idea where one forgets about efficiency and carries out attention on each edge of the nodes (tokens). You can think of it as doing attention on the attention matrix, taking the perspective of the attention matrix as all the directed edges of a fully connected graph.

The hypothesis is that there is a task out there that the (sub)quartic transformer can do that quadratic transformers cannot.

## Todo

- [x] first add a weak taylor linear attention on top of all edges

- [ ] use coordinate descent routing from the node attention matrix to select a subset of edges to update (and do full attention across)

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
@misc{Sutton,
    title  = {The Bitter Lesson},
    url    = {http://www.incompleteideas.net/IncIdeas/BitterLesson.html},
    author = {Sutton, Rich}
}
```

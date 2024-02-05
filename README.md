## Quartic Transformer (wip)

Exploring an idea where one forgets about efficiency and carries out attention on each edge of the nodes (tokens)

## Todo

- [ ] first add a weak taylor linear attention on top of all edges
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

# Leveraging Knowledge Bases in LSTMs for Improving Machine Reading

Code and data for the paper "Leveraging Knowledge Bases in LSTMs for Improving Machine Reading" by Bishan Yang and Tom Mitchell.

If you are using this code or data in your work, please cite the following publication:

Bishan Yang and Tom Mitchell (2017). Leveraging Knowledge Bases in LSTMs for Improving Machine Reading. Association for Computational Linguistics (ACL), 2017.

For any questions/issues about the code, please feel free to email Bishan Yang (bishan.yang@gmail.com).

```
@InProceedings{yang2017kblstm,
  author    = {Yang, Bishan and Mitchell, Tom},
  title     = {Leveraging Knowledge Bases in LSTMs for Improving Machine Reading},
  booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year      = {2017}
}
```

# Code
The code was implemented and tested using Theano (0.9.0). 

# Data
`nell_concept2vec.txt` stores the pretrained 100-dim embeddings for concepts in the ontology of NELL (<http://rtw.ml.cmu.edu/rtw>).
`wn_concept2vec.txt` stores the pretrained 100-dim embeddings for synsets in WordNet.

# References

tagger: Named Entity Recognition Tool (<https://github.com/glample/tagger>)

# License

Apache-2.0
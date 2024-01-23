# OWL2Vec-Star Extended version 


OWL2Vec-Star Extended version 2023 by Sevinj Teymurova


[![pypi](https://img.shields.io/pypi/v/owl2vec-star.svg)](https://pypi.python.org/pypi/owl2vec-star)
[![docs](https://readthedocs.org/projects/owl2vec-star/badge/?version=latest)](https://owl2vec-star.readthedocs.io/en/latest/?version=latest)


## **OWL2Vec*Ext: Embedding OWL ontologies**


### Features
--------

After installation, OWL2Vec*Ext exposes a CLI with two subcommands that enable the execution of two main programs. Additionally, you have the option to run the two original Python programs without the need for installation. (see the requirements in `setup.py <https://github.com/KRR-Oxford/OWL2Vec-Star/blob/master/setup.py>`__).

Installation command:
```
    $ make install
```
### Standalone
------------

This command will embed two ontologies along with their corresponding mappings. Configuration can be customized using the default1.cfg file. Refer to the examples and comments in default1.cfg for usage guidance.


### Running program::
------------

#### Option 1

1. Install Python 3: https://python.org/downloads
2. Install setuptools: https://pypi.org/project/setuptools
3. Run this command in the terminal: 
```
    $ python setup.py install
```

4. Run this command in the terminal:
```
    $ python OWL2Vec_Standalone1.py --config_file default1.cfg
```

#### Option 2

1. Install Python 3: https://python.org/downloads
2. Install pip: https://pip.pypa.io/en/stable/installation
3. Install library dependicies in requirements: pip install -r requirements_owl2vec.txt
4. Run jupyter notebook: 
``` 
    jupyter_notebook_owl2vec_star_ext.ipynb 
```

Note: In contrast to the experimental codes, the standalone command has implemented all OWL ontology-relevant procedures in Python with Owlready. However, it also enables users to utilize pre-calculated annotations, axioms, entities, or projections to generate the corpus of sentences. 

**Parameters to change when running the code with help of configuration file**
1. ontology_file1 
2. ontology_file2 
3. mapping
4. confidence_threshold
5. seed_entities_on_mapping
6. cache_dir
7. walk_depth



## Publications


### Main Reference
------------


- Jiaoyan Chen, Pan Hu, Ernesto Jimenez-Ruiz, Ole Magnus Holter, Denvar Antonyrajah, and Ian Horrocks. **OWL2Vec*: Embedding of OWL ontologies**. Machine Learning, Springer, 2021. [PDF](https://arxiv.org/abs/2009.14654) [Springer](https://rdcu.be/cmIMh) [Collection](https://link.springer.com/journal/10994/topicalCollection/AC_f13088dda1f43d317c5acbfdf9439a31) [Codes in package](https://github.com/KRR-Oxford/OWL2Vec-Star/releases/tag/OWL2Vec-Star-ML-2021-Journal) or [folder](https://github.com/KRR-Oxford/OWL2Vec-Star/tree/master/case_studies)



### Applications with OWL2Vec*Ext
------------

- Jiaoyan Chen, Ernesto Jimenez-Ruiz, Ian Horrocks, Denvar Antonyrajah, Ali Hadian, Jaehun Lee. **Augmenting Ontology Alignment by Semantic Embedding and Distant Supervision**. European Semantic Web Conference, ESWC 2021. [PDF](https://openaccess.city.ac.uk/id/eprint/25810/1/ESWC2021_ontology_alignment_LogMap_ML.pdf) [LogMap Matcher work](https://github.com/ernestojimenezruiz/logmap-matcher/)
- Ashley Ritchie, Jiaoyan Chen, Leyla Jael Castro, Dietrich Rebholz-Schuhmann, Ernesto Jiménez-Ruiz. **Ontology Clustering with OWL2Vec***. DeepOntonNLP ESWC Workshop 2021. [PDF](https://openaccess.city.ac.uk/id/eprint/25933/1/OntologyClusteringOWL2Vec_DeepOntoNLP2021.pdf)



### Preliminary Publications
------------

- Ole Magnus Holter, Erik Bryhn Myklebust, Jiaoyan Chen and Ernesto Jimenez-Ruiz. **Embedding OWL ontologies with OWL2Vec**. International Semantic Web Conference. Poster & Demos. 2019. [PDF](https://www.cs.ox.ac.uk/isg/TR/OWL2vec_iswc2019_poster.pdf)
- Ole Magnus Holter. **Semantic Embeddings for OWL 2 Ontologies**. MSc thesis, University of Oslo. 2019. [PDF](https://www.duo.uio.no/bitstream/handle/10852/69078/thesis_ole_magnus_holter.pdf) [GitLab](https://gitlab.com/oholter/owl2vec)



### Case Studies
------------
Data and codes for class membership prediction on the Healthy Lifestyles (HeLis) ontology, as well as class subsumption prediction on the food ontology FoodOn and the Gene Ontology (GO), [MyUniversity], and [Conference] are located under the folder case_studies/.`.


### Credits
-------
The code located at "owl2vec_star/rdf2vec/mapping4vec.py" implements a biased sampling walking strategy over RDF graphs (version 0.0.1, last accessed: 08/2023) with revisions.

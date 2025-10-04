# KGR4VRD: Knowledge Graph Reasoning for Visual Relationship Detection

## KGR4VRD
The **KGR4VRD** repository hosts code associated with aspects of our **neurosymbolic AI research**. This research explores combining symbolic knowledge and symbolic reasoning with deep (subsymbolic) learning in neurosymbolic systems directed at the computer vision task of detecting visual relationships in images. An example visual relationship for a hypothetical image is, say, **(person, ride, horse)**. 

For the symbolic elements (knowledge and reasoning) of our neurosymbolic systems, we rely entirely on **Semantic Web** technologies. That is, to represent symbolic knowledge, we use **OWL** (the Web Ontology Language), **OWL ontologies** (descriptions of domains of interest that have associated inference semantics), and **OWL knowledge graphs** (ontologies together with asserted facts in RDF triple stores).  For symbolic reasoning, we use **OWL reasoning**, or OWL knowledge graph reasoning. Since OWL is the Semantic Web incarnation of the formal Description logic $\mathcal{SROIQ}$, OWL reasoning is really $\mathcal{SROIQ}$ reasoning. Hence, it is guaranteed to be both logically sound and complete. Description logics are decidable fragments of first-order logic, and $\mathcal{SROIQ}$ is a particularly expressive fragment.

We describe this research in the (forthcoming) paper:
> **Herron, D., Jimenez-Ruiz, E., Weyde, T.** (2026). `Logical Reasoning with OWL Knowledge Graphs for Learning and Validating Visual Relationship Detection.` (under review)


## NeSy4VRD

Our research, and hence our code, uses the NeSy4VRD dataset.
The acronym NeSy4VRD stands for "Neurosymbolic AI for Visual Relationship Detection". 
NeSy4VRD is a unique `image dataset plus OWL ontology' resource that we created to facilitate our neurosymbolic AI research.

The NeSy4VRD dataset package contains *(i)* the NeSy4VRD dataset, consisting of an image dataset with high-quality visual relationship annotations, and *(ii)** the NeSy4VRD OWL ontology, called **VRD-World**.







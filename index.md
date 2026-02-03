---
layout: project_page
permalink: /

# You can declare accent colors here
# accent: '#D21111'
# accent: darkorange

title: Concepts' Information Bottleneck Models
authors:
    - name: Karim Galliamov
      link: https://scholar.google.com/citations?user=tBhN7ecAAAAJ&hl=en
      affiliation: 1
    - name: Syed M Ahsan Kazmi
      link: https://people.uwe.ac.uk/Person/AhsanKazmi
      affiliation: 2
    - name: Adil Khan
      link: https://www.hull.ac.uk/staff-directory/adil-khan
      affiliation: 3 
    - name: Adín Ramírez Rivera
      link: https://www.mn.uio.no/ifi/english/people/aca/adinr/
      affiliation: 4
affiliations:
    - name: University of Amsterdam
      link: https://www.uva.nl/en
    - name: University of the West of England, Bristol
      link: https://people.uwe.ac.uk
    - name: University of Hull
      link: https://www.hull.ac.uk
    - name: University of Oslo
      link: https://uio.no
paper: https://openreview.net/forum?id=JGIYfwaNpT
#video: https://www.youtube.com/@UniOslo
code: "{{ site.github.repository_url }}" # in case you want to use the same repo where the gh-pages is (the most common setup)
# code: https://github.com/dsb-ifi # in case you want to hard-code the repo
# data: https://huggingface.co/docs/

abstract: Concept Bottleneck Models (CBMs) aim to deliver interpretable predictions by routing decisions through a human-understandable concept layer, yet they often suffer reduced accuracy and concept leakage that undermines faithfulness. We introduce an explicit Information Bottleneck regularizer on the concept layer that penalizes $I(X;C)$ while preserving task-relevant information in $I(C;Y)$, encouraging minimal-sufficient concept representations. We derive two practical variants (a variational objective and an entropy-based surrogate) and integrate them into standard CBM training without architectural changes or additional supervision. Evaluated across six CBM families and three benchmarks, the IB-regularized models consistently outperform their vanilla counterparts. Information-plane analyses further corroborate the intended behavior. These results indicate that enforcing a minimal-sufficient concept bottleneck improves both predictive performance and the reliability of concept-level interventions. The proposed regularizer offers a theoretic-grounded, architecture-agnostic path to more faithful and intervenable CBMs, resolving prior evaluation inconsistencies by aligning training protocols and demonstrating robust gains across model families and datasets.

---

![Diagram summarizing the method]({{ site.image_base_path | append: "teaser.png" | absolute_url }})

{: .figure-caption}
**Our proposed CIBMs pipeline.** The image is encoded through $p(z \mid x)$, which in turn encodes the concepts with $q(c \mid z)$, and the labels are predicted through $q(y \mid c)$.  These modules are implemented as neural networks.  We introduced the IB regularization as mutual information optimizations over the variables as shown in dashed lines.


## Citation
{% raw %}
```
@inproceedings{galliamov2026,
title={Concepts' Information Bottleneck Models},
author={Galliamov, Karim and Kazmi, Syed M Ahsan and Khan, Adil and Ram\'irez Rivera, Ad\'in},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=JGIYfwaNpT}
}
```
{% endraw %}

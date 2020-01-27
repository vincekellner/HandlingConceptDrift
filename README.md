# HandlingConceptDrift
Strategies to detect and handle incremental concept drift in time series data

Based on the dataset of NYC Taxi and Limousine Commission (TLC) (https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page) in which various types of drift can be detected, different strategies are provided how to handle incremental drift. The general idea is to provide different approaches to handle incremental drift even though one does not known in advance whether the data is affected by drift or not.

## Definition of concept drift

Concepts in the real world are often not stable and might change over time. Especially in nonstationary dynamic environments the underlying distribution of data can change over time. This effect is referred to as concept drift in literature (Widmer and Kubat 1996) and makes the creation of a model only based on past data challenging since the model is inconsistent with new data (Tsymbal 2004, p.1).  
Formally concept drift is defined as follows: 
∃𝑋:𝑝𝑡0(𝑋,𝑦) ≠ 𝑝𝑡1(𝑋,𝑦) 

The joint distribution between the input X and target y, denoted by 𝑝𝑡0 at time t0, differs from the distribution at t1. Changes might occur in different ways: 
 Prior probabilities of the target 𝑝(𝑦) may change. 
 Conditional probabilities 𝑝(𝑋|𝑦) may change. 
(Gama et al. 2014, pp.4; Žliobaité 2010, pp.3)

Gama et al. (2014) define different forms, the data distribution can change over time: Drift may happen abruptly such that the data switched suddenly to another concept, incrementally where slightly different concepts change slowly, or gradually where two concepts are active and keep switching back and forth for some time while the probability to obtain data from the first concept decreases. Concepts might also reoccur after some time. Reoccurring concepts differ from periodic seasonality since they happen without certainty (Gama et al. 2014, pp.11). 

Learning algorithms that provide the ability to adapt to changes in the data-generating process are referred to as adaptive learning algorithms (Gama et al. 2014, p.3). 

The importance of addressing the challenges of concept drift increases, since more and more applications organize data in a data stream format rather than in a static database to apply models online. In this streaming setting, it is most likely that the data distribution will change over time. Consequently, instead of monitoring the performance of the application and adjusting or retraining a deployed model manually every time it becomes outdated, the focus is shifted towards more automation of the model development and updating tasks (Žliobaité, Pechenizkiy and Gama 2016, p.92). 



## Adaptation strategies 

All adaptation strategies except for the “Switching Scheme” are inspired by the taxonomy of adaptive learning systems proposed by Gama et al. (2014). Gama et al. (2014) distinguishes between the learning mode of a model and adaptation methods. The learning mode refers to how a model is updated if new data is available. Two learning modes are proposed: 1) Regular retraining of the model and discarding of old models and 2) incremental updating of the model based on the most recent observations. Concerning adaptation methods Gama et al. (2014) distinguishes whether the model is adapted based on a trigger such as a change detector or adapted periodically without any explicit detection of change (Gama et al. 2014, pp.11). 

#### Blind adaptation strategies

The regular adaptation class refers to the idea to adapt models periodically based on some specified frequency (Gama et al. 2014, pp.11).  Two learning modes are distinguished:

- Regular training of a new model

- Regular incremental training of a model

#### Informed adaptation strategies with drift detectors

The triggered adaptation strategies refer to the idea to initiate a model update or retraining based on explicit drift detection as proposed by Gama et al. (2014). Incoming data is monitored on a continuous basis and statistical tests are performed to detect drift. If a change is suspected, an adaptive action such as a retraining is triggered (Gama et al. 2014, pp.11). 

- Regular training of a new model triggered by a detection mechanism

- Regular incremental training of a model triggered by a detection mechanism

- Combination of incremental trainings & training of a new model triggered by a detection mechanism ("Switching Scheme")


## The "Switching Scheme" a novel approach to handle incremental drift
The idea behind this novel adaptation scheme is to take advantage of the individual benefits of a complete retraining and an incremental updating strategy. The initial model is kept and is incrementally updated with the most recent observations as long as possible. If the model seems to be outdated or does not adapt fast enough to a new concept, a new model is trained and the old model is discarded. 
In this work, a time frame τ is specified for how long the model is incrementally updated if drift is detected before a new model is trained. After a retraining, the new model is incrementally updated if drift is detected until the next τ is reached.


## References


Gama, João / Žliobaité, Indré / Bifet, Albert / Pechenizkiy, Mykola / Bouchachia, Abdelhamid (2014), A survey on concept drift adaptation, ACM Computing Survey (CSUR), Vol. 46, Issue 4, Article 44, 37 pages.

Tsymbal, Alexey (2004), The Problem of Concept Drift: Deﬁnitions and Related Work, (Technical Report TCD-CS-2004-15), Department of Computer Science, Trinity College, Dublin, 7 pages.

Widmer, Gerhard / Kubat, Miroslav (1996), Learning in the Presence of Concept Drift and Hidden Contexts, Machine Learning, Vol. 23, Issue 1, pp.69-101. 

Žliobaité, Indré / Pechenizkiy, Mykola / Gama, João (2016), An Overview of Concept Drift Applications, Japkowicz, Nathalie / Stefanowski, Jerzy (Eds.), Big Data Analysis: New Algorithms for a New Society, Studies in Big Data, Volume 16, Springer International Publishing Switzerland 2016, pp.91-114. 

Žliobaité, Indré, (2010), Learning under Concept Drift: an Overview. Technical report, Vilnius University.

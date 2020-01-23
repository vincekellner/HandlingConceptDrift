# HandlingConceptDrift
Strategies to detect and handle incremental concept drift in time series data

Based on the dataset of NYC Taxi and Limousine Commission (TLC) (https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page) in which various types of drift can be detected, different strategies are provided how to handle incremental drift.


## Definition of concept drift



## Adaptation strategies 


#### Blind adaptation strategies:

- Regular training of a new model

- Regular incremental training of a model

#### Informed adaptation strategies with drift detectors:

- Regular training of a new model triggered by a detection mechanism

- Regular incremental training of a model triggered by a detection mechanism

- Combination of incremental trainings & training of a new model triggered by a detection mechanism ("Switching Scheme")


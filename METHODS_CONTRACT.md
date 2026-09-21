# Methodological contract


The frozen expression model uses a diagonal-Gaussian encoder, a softmax transform and a row-normalized topic-gene decoder. Its loss is equal-cell cross-entropy on panel proportions plus 0.01 times Gaussian KL to a standard normal. It is a logistic-normal regularized representation, not an exact Dirichlet model, an unmodified count-likelihood ELBO or a calibrated classifier of biological states. Deterministic projection uses softmax of the Gaussian mean, not the posterior mean of the softmax.

Three estimands must remain separate: mean topic loading within gated fibroblasts, the fraction of those fibroblasts above a loading cut, and the mean topic loading over all retained cells from a mapped patient. The latter is the primary healing contrast. Repeated specimens are pooled by retained cell count within patient; patients then receive equal weight within each outcome arm. The outcome subset has 11 patients, not 14 independent specimens or thousands of independent cells.

The threshold mean identity is exact even when one bin is empty, provided its weighted contribution is zero. A correlation between the mean and threshold fraction does not imply constant bin means, interchangeable estimands, two biological attractors or a causal abundance effect. The full fitted cut 0.184295848 and historical rounded technical-control cut 0.184 have distinct documented uses. The new frozen-score analysis measures the residual of a stated constant-bin approximation and excludes undefined empty-bin means from their ranges.

The historical patient permutation statistic used add-one correction over all 330 allocations. Both that recorded probability, 90/331, and the exhaustive fraction, 89/330, are reported. Their interpretation assumes observational label exchangeability. The 90% Welch interval and two one-sided equivalence tests agree algebraically; their data-derived boundary does not supply a clinical equivalence margin. Fixed-score AUC bootstraps exclude learning-algorithm uncertainty. Discovery representation fitting includes expression from subsequently held-out patients, so the discrimination comparison is internal.


Historical report fingerprints describe the producing computation. Current source includes documentation corrections and input-domain guards; it is not retroactively attributed to earlier runs. The release manifest identifies this exact distributed version.

# Mathematical Formulation — NOVA-Loc

## 1. Coordinate system

Search image: `S ∈ [0,1]^(H×W)` with `H=W=1000` in the challenge scaffold.

Reference image: `R ∈ [0,1]^(h×w)` with the challenge nominal relationship `h ≈ 10·h_s`, `w ≈ 10·w_s` for the true search-site crop.

Let the search-space center be

`c = ((W-1)/2, (H-1)/2)`.

## 2. Candidate generation

For scale ratio `r` near `1/10`, resize the reference:

`T_r = Resize(R, r)`.

Compute ZNCC:

`ρ_r(x,y) = ZNCC(T_r, S[x:x+h_r, y:y+w_r])`.

Keep the top K local maxima across all r.

## 3. Local structure

Let `G(I)` be normalized gradient magnitude and optionally an oriented-gradient descriptor. Define local gradient similarity `G_i` between the resized reference and the candidate patch.

## 4. Context ring

Let `B_i` be a rectangular window around candidate i and let `R_i` be the corresponding inner crop. The context ring is `B_i \ R_i`. Define a ring score

`Q_i = NCC(Context(R), Context(B_i))`.

The goal is to distinguish repeated local motifs whose neighborhoods differ.

## 5. Periodicity

For an image `I`, compute the magnitude spectrum

`P_I(u,v) = |FFT2(I)|`.

Estimate dominant periodicity peaks excluding the DC neighborhood. A candidate receives a compatibility score `F_i` based on agreement between reference and candidate peak locations / autocorrelation spacing.

## 6. Ambiguity

Let candidate scores before final selection be `J_(1) ≥ J_(2)`. Define

`U = (J_(1)-J_(2))/(|J_(1)|+ε)`.

Small `U` means multiple candidates are almost equally plausible and activates the deep verification path.

## 7. Final score

`J_i = wC Ci + wQ Qi + wG Gi + wF Fi + wS Si - wA Ai - λ Di`.

Here `D_i` is normalized center distance. The center prior is intentionally used late in the pipeline because it is a rule from the problem specification, not a substitute for visual matching.

## 8. Error metrics

Euclidean localization error:

`e_i = sqrt((xhat_i-x_i)^2 + (yhat_i-y_i)^2)`.

Tolerance accuracy at τ pixels:

`Acc(τ) = (1/N) Σ 1[e_i ≤ τ]`.

We recommend reporting τ ∈ {1,2,5,10} px.

## 9. OOD robustness

Define an OOD degradation ratio

`R_OOD = Acc_OOD / Acc_ID`.

Also report the absolute accuracy gap `Δ = Acc_ID - Acc_OOD`.

## 10. Runtime

Measure end-to-end inference on a 1000×1000 pair, excluding one-time process startup if and only if the benchmark specification allows it. Report mean and p95 over at least 30 pairs.
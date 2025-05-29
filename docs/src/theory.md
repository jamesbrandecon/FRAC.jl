# Fast, Robust and Approximately Correct (FRAC) Estimation of the BLP Model

## 1. Background: The Standard BLP Framework

Berry, Levinsohn & Pakes (1995) propose a random‐coefficients logit model of differentiated‐product demand.  Key objects:

* **Market shares:** $s_{jt}$ for product *j* in market *t*, and an outside good share $s_{0t}$.
* **Mean individual-level utilities:** $\delta_{ijt} = X_{jt}^\top\beta + \xi_{jt} + \Sigma_k \sigma_k x_{jkt} v_{ikt}$, with idiosyncratic shocks $\varepsilon_{ijt}.$
* **Endogeneity:** Prices (and possibly other characteristics) are correlated with unobserved quality $\xi_{jt}.$

BLP solves a fixed point problem to invert shares to mean utilities $\delta_{jt}$, then estimates $\beta, \sigma$ by GMM using instruments for endogenous covariates $Z_{jt}.$  This is powerful but **computationally expensive** because each objective‐function evaluation requires many evaluations of the contraction mapping, and because that contraction requires integrating over potentially many dimensions of random coefficients.

---

## 2. FRAC: A Second-Order Approximation

FRAC sidesteps repeated inversions by approximating the *individual* share function around the population mean of the random coefficients.  A second-order Taylor expansion yields

$$
\log\!\left(\frac{s_{jt}}{s_{0t}}\right)
\;\approx\; X_{jt}^\top\beta\; +\; \underbrace{\tfrac12\,X_{jt}^\top\Sigma X_{jt}}_{\text{quadratic term}} \; +\; e_{jt}^\top\xi_{t} + \varepsilon_{jt},
$$

where the quadratic term can be written as a linear combination of **artificial regressors** $K_{mn}^{jit}$ constructed from product characteristics.

Crucially, after augmentation the model becomes **linear in the unknown parameters** $\beta$ and the non-zero elements of $\Sigma$.  Estimation therefore reduces to (instrumented) least-squares rather than nested fixed-point GMM.

---

## 3. Estimation via Algorithm 1 from Salanie & Wolak (2022)

1. **Construct dependent variable**
   For each market *t*, stack the original $(s_{1t},\ldots,s_{Jt})$ with the outside share to obtain $(S_{0t},S_{1t},\ldots,S_{Jt}).$

2. **Form regressors for every product–market pair $(j,t)$**

   1. Compute the **market-share-weighted means** of characteristics:
      $e_t = \sum_{k=1}^{J} S_{kt} X_{kt}.$
   2. For every index pair $(m,n)$ in the set of unique characteristic pairs $\mathcal I$:

      * *Diagonal* ($m=n$):
        $K^{jit}_{mm} = \bigl( x_{jtm}/2 - e_{tm} \bigr) X_{jtm}.$
      * *Off-diagonal* ($n<m$):
        $K^{jit}_{mn} = X_{jtm} X_{jtn} - e_{tm} X_{jtn} - e_{tn} X_{jtm}.$
   3. Define the *dependent variable*
      $y_{jt} = \log(S_{jt}/S_{0t}).$

3. **Two-Stage Least Squares (2SLS)**
   Regress $y$ on the stacked matrix $[X\;K]$, instrumenting each column with flexible functions of the excluded instruments $Z$.  Denote the coefficient vector as $\widehat\beta$ (elements on *X*) and $\widehat\Sigma$ (non-zero elements on *K*).

4. **(Optional) Three-Stage Least Squares (3SLS)**
   To pool information across markets, run a 3SLS over the stacked *J* equations, using the residual variance–covariance from Step 3 as a weighting matrix.

Because every step is linear, **computation is orders of magnitude faster** than full BLP estimation, with negligible loss in accuracy for moderate random-coefficient variance.

---

## 4. Why Debias?  The Bootstrap Correction

The second-order expansion neglects higher-order terms, so $\widehat\Sigma$ is *approximately* unbiased only when random-coefficient variation is small.  To correct the remaining bias Fox *et al.* propose a **parametric bootstrap**:

1. **Simulate artificial markets** using the fitted parameters $(\widehat\beta, \widehat\Sigma)$ and the original instruments and characteristics.  Draw individual tastes $v_i$ from their assumed distribution, compute true BLP shares, and generate “pseudo‐data.”
2. **Re-estimate FRAC** on each bootstrap draw to obtain $\widehat\theta^{*(b)}.$
3. **Bias‐correct** the original estimator:

   $$
   \theta^C \;=\; 2\,\widehat\theta \;-\;\frac{1}{B}\sum_{b=1}^B \widehat\theta^{*(b)},
   $$

   where $\theta^C$ is the bias‐corrected parameter vector.

4. **Inference:** Use the distribution of $\widehat\theta^{*(b)} - \overline{\widehat\theta^{*}}$ to construct confidence intervals that account for both sampling error and approximation error.

In practice 200–500 bootstrap replications suffice, and because each replication is just another FRAC run, the full bias-corrected procedure is still dramatically faster than one iteration of the nested fixed-point estimator.

---

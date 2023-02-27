---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: venv
    language: python
    name: venv
---

# Mixtures & bends

```python tags=[]
%load_ext autoreload
%autoreload 2
```

```python tags=[]
import numpy as np
import scipy.stats
import pandas as pd

import jax
import jax.numpy as jnp
import jax.scipy.special as jsc
import optax

from tqdm.auto import tqdm
import plotly.express as px
import plotly.graph_objects as go
import colorcet
import plotly_template
```

```python tags=[]
import data as dg
import block_mt as bmt
```

```python tags=[]
dist = dg.mixture(
    (
        scipy.stats.multivariate_normal(
            (2., 1.),
            ((1., .9),
             (.9, 1.))
        ),
        scipy.stats.multivariate_normal(
            (1., 2.),
            ((1., -.9),
             (-.9, 1.))
        ),
    ),
    (.5, .5)
)

n_samples = 200
dims = ['x', 'y']
n_dims = len(dims)
df = dg.mixture_to_df(*dist.rvs(size=n_samples, random_state=1), ('A', 'B', 'C'), dims)

data_fig = (
    px.scatter(df, x='x', y='y')
    .update_traces(marker={'color': 'black', 'size': 4})
    .update_yaxes(scaleanchor='x', scaleratio=1)
)
data_fig
```

```python tags=[]
Lx = np.linspace(df.x.min() - .5, df.x.max() + .5)
Ly = np.linspace(df.y.min() - .5, df.y.max() + .5)
```

```python tags=[]
data = np.array(df[dims]).T
```

```python tags=[]
(
    px.scatter(df, x='x', y='y', color='label')
    .update_yaxes(scaleanchor='x', scaleratio=1)
)
```

```python tags=[]
def product_pll(log_precs):
    """
    Product quasi-likelihood
    """
    prod_mean, prod_prec = 0., 0.
    for log_prec in log_precs:
        means, variances, _log_pdfs = (
            bmt.NonCentralMatrixT
            .from_params(
                .5, left=jnp.eye(n_dims),
                right=jnp.diag(jnp.exp(-log_prec)),
                gram_mean_left=0.5
            )
            .observe(data)
            .uni_cond()
        )
        prod_prec += 1. / variances
        prod_mean += means / variances
    prod_mean /= prod_prec
    prod_log_p = jnp.sum(
        - 0.5 * jnp.log(jnp.pi)
        + 0.5 * jnp.log(prod_prec)
        - 0.5 * prod_prec * (data - prod_mean)**2
    )
    return prod_log_p       
```

```python tags=[]
@jax.jit
def nppelbo(log_precs, prior_log_alphas):
    """
    Negative product pseudo evidence lower bound
    
    log_precs: (n_components, n_samples)
    
    """
    # Mnemonics:: k: components, n:samples, d: dims
    log_precs_kn = log_precs ; del log_precs
    prior_log_alphas_k = prior_log_alphas ; del prior_log_alphas
    data_nd = data
    
    # Bijectors
    prior_alphas_k = jnp.exp(prior_log_alphas_k)
    precs_kn = jnp.exp(log_precs_kn)

    # Analytical optimizations
    var_alphas_k = prior_alphas_k + .5 * n_dims
    prior_scales_k = jnp.mean(precs_kn, -1) / prior_alphas_k

    # Product pseudo likelihood at variational mean
    ppe = product_pll(log_precs_kn)
    
    # Prior at variational mean
    log_prior_mean_kn = jax.scipy.stats.gamma.logpdf(
        precs_kn,
        prior_alphas_k[:, None],
        scale=prior_scales_k[:, None],
    )
    # Variational dist at variational mean
    log_var_mean_kn = jax.scipy.stats.gamma.logpdf(
        precs_kn,
        var_alphas_k[:, None],
        scale=precs_kn/var_alphas_k[:, None]
    )
    return (
        - ppe
        - jnp.sum(log_prior_mean_kn)
        + jnp.sum(log_var_mean_kn)
    )
```

```python tags=[]
n_cps = 2
opt = optax.adam(0.2)
rng = np.random.default_rng(0)
params = dict(
    log_precs = jnp.array(
        rng.normal(0., 1., size=(n_cps, n_samples)),
        jnp.float32),
    prior_log_alphas = jnp.zeros((n_cps,), jnp.float32),
)
state = opt.init(params)
trace = []
for i in tqdm(range(200)):
    val, grad = jax.value_and_grad(lambda kw: nppelbo(**kw))(params)
    trace.append({
        'iteration': i,
        'nppelbo': val,
        ** {
        f'prior_log_alpha_{k}': pla
            for k, pla in enumerate(params['prior_log_alphas'])
        }
    })
    dp, state = opt.update(grad, state)
    for k in params:
        params[k]  += dp[k]

trace = pd.DataFrame(trace)
px.line(trace, x='iteration', y='nppelbo')
```

```python tags=[]
for i in (0, 1):
    trace[f'prior_alpha_{i}'] = np.exp(trace[f'prior_log_alpha_{i}'].astype(np.float32))
```

```python tags=[]
px.line(trace, x='iteration', y=[f'prior_alpha_{i}' for i in (0,1)], log_y=True)
```

```python tags=[]
opt_df = pd.concat((
    pd.DataFrame(params['log_precs'].T, columns=['u', 'v']),
    df
), axis=1)
opt_df['affinity'] = 1. / (1. + np.exp(opt_df.v - opt_df.u))
px.scatter(
    opt_df,
    x='u', y='v',
    color='label'
)
```

```python tags=[]
px.scatter(
    opt_df,
    x='x', y='y',
    color='affinity',
    color_continuous_scale=colorcet.CET_D4
)
```

```python tags=[]
posts = [
    bmt.NonCentralMatrixT
    .from_params(
        .5, left=jnp.eye(n_dims),
        right=jnp.diag(jnp.exp(-log_prec)),
        gram_mean_left=0.5
    )
    .observe(data)
    .post()
    for log_prec in params['log_precs']
]

@jax.jit
def _logk_prec(new, log_precs):
    print('Tracing...')
    return sum([
        post
            .extend(jnp.exp(-log_prec) * np.eye(1))
            .observe(new[:, None])
            #.log_pdf()
            .uni_cond()[-1].sum()
        for post, log_prec in zip(posts, log_precs)
    ])
logPs = jnp.stack([
    jax.vmap(lambda x: jax.vmap(lambda y: _logk_prec(jnp.array([x, y]), lp))(Ly))(Lx)
     for lp in tqdm(params['log_precs'].T)])
```

```python tags=[]
logPs.shape
```

```python tags=[]
avg_logP = jsc.logsumexp(logPs, 0) - jnp.log(n_samples)
```

```python tags=[]
dLx = Lx[1] - Lx[0]
dLy = Ly[1] - Ly[0]
Z = jnp.exp(jsc.logsumexp(avg_logP))*dLx*dLy
print(Z)
```

```python tags=[]
go.Figure(data_fig).add_contour(
    x=Lx, y=Ly, z=avg_logP - np.log(Z), #zmin=0., zmax=0.01,
    contours={'coloring': 'heatmap'}, ncontours=10, colorscale=colorcet.CET_L18,
    transpose=True)
```

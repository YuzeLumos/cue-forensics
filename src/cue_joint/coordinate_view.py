"""Exact coordinates of the SAME constrained estimator, not a new candidate."""
import numpy as np


def coordinate_readout(core):
    if core.current is None:
        raise ValueError('No current support: use the exact zero-correction anchor path')
    cells=core.current.cells;n=sum(c.n for c in cells)
    mean=sum((c.sum for c in cells),np.zeros(257))/n
    contrast=(cells[1].sum/cells[1].n-cells[0].sum/cells[0].n)/2
    S=sum((c.scatter() for c in cells),np.zeros((257,257)))
    H,nu=core.historical_scatter();W=(S+H)/(n+nu);W=(W+W.T)/2
    R=core.basis
    if R.shape[1]==0:
        return dict(U=np.zeros((256,0)),beta=np.zeros(0),weight=np.zeros(256),bias=-float(mean[-1]),whitening_error=0.)
    G=R.T@(W[:256,:256]+core.lambda_prior/n*core.metric)@R
    L=np.linalg.cholesky(G)
    U=np.linalg.solve(L,R.T).T  # R L^{-T}
    d=U.T@contrast[:256];q=U.T@W[:256,256]
    beta=np.linalg.solve(np.eye(len(d))+np.outer(d,d),(1-contrast[256])*d-q)
    weight=U@beta;bias=float(-mean[256]-mean[:256]@weight)
    white=U.T@(W[:256,:256]+core.lambda_prior/n*core.metric)@U
    return dict(U=U,beta=beta,weight=weight,bias=bias,whitening_error=float(np.max(np.abs(white-np.eye(len(d))))))

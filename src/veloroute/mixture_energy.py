"""Distribution supervision for a router over fixed candidate endpoints."""
import torch


def mixture_energy(candidates, probabilities, target, *, detach_candidates=True):
    """Exact energy V-statistic of the conditional discrete mixture.

    Fixed-candidate probes default to router-only gradients. Joint training sets
    detach_candidates=False to update endpoints too. Targets remain constants.
    This objective does not identify individual cell fates.
    """
    if candidates.ndim != 3 or not len(candidates) or probabilities.shape != candidates.shape[:2]:
        raise ValueError('Candidate/probability dimensions mismatch')
    if target.ndim != 2 or target.shape[1] != candidates.shape[-1] or not len(target):
        raise ValueError('Target dimensions mismatch')
    if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError('Invalid mixture probabilities')
    if not torch.allclose(probabilities.sum(-1), probabilities.new_ones(len(probabilities)), atol=1e-5):
        raise ValueError('Mixture probabilities must sum to one per source')
    if not torch.isfinite(candidates).all() or not torch.isfinite(target).all():
        raise ValueError('Nonfinite mixture endpoints/targets')
    points = (candidates.detach() if detach_candidates else candidates).flatten(0, 1)
    target = target.detach()
    weights = probabilities.flatten()/len(probabilities)
    cross = torch.cdist(points, target).mean(-1)
    within = torch.cdist(points, points)
    return 2*(weights*cross).sum()-weights@within@weights-torch.cdist(target, target).mean()


@torch.no_grad()
def candidate_endpoints(model, state, condition, *, start=4., end=5., step=.25):
    """Exact existing RK4 policy, for frozen fields without intrinsic/gate/noise."""
    if model.config.use_intrinsic or model.config.use_noise or model.config.use_gate:
        raise ValueError('Fixed-candidate router experiment forbids extra dynamic branches')
    count = int(round((end-start)/step))
    if count < 1 or abs(count*step-(end-start)) > 1e-8:
        raise ValueError('Invalid physical RK4 grid')
    values = []
    for k in range(model.config.max_experts):
        mode = torch.full((len(state),), k, dtype=torch.long, device=state.device)
        z = state.clone()
        for index in range(count):
            t = start+index*step
            a = model.field.selected(z, t, condition, mode)
            b = model.field.selected(z+step*a/2, t+step/2, condition, mode)
            c = model.field.selected(z+step*b/2, t+step/2, condition, mode)
            d = model.field.selected(z+step*c, t+step, condition, mode)
            z = z+step*(a+2*b+2*c+d)/6
        values.append(z)
    return torch.stack(values, 1)

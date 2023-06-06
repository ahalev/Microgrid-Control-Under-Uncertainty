from torch.distributions.kl import register_kl, kl_divergence
from garage.torch.distributions import TanhNormal


def register_tanhnormal():
    @register_kl(TanhNormal, TanhNormal)
    def kl_tanh_normal(p, q):
        # Not mathematically correct, but should work in practice
        return kl_divergence(p._normal, q._normal)

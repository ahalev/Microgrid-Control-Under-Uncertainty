from reward_shaping.base import BaseRewardShaper


class PVCurtailmentShaper(BaseRewardShaper):
    yaml_tag = u"!PVCurtailmentShaper"

    def __call__(self, step_info, cost_info):
        try:
            pv_info = step_info['pv']
        except KeyError:
            return 0.0
        else:
            curtailment = [d['curtailment'] for d in pv_info]
            return -1.0 * sum(curtailment)

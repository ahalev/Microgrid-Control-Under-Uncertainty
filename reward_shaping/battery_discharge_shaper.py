from reward_shaping.base import BaseRewardShaper


class BatteryDischargeShaper(BaseRewardShaper):
    yaml_tag = u"!BatteryDischargeShaper"

    def __call__(self, step_info, cost_info):
        pass

from reward_shaping.base import BaseRewardShaper


class BatteryDischargeShaper(BaseRewardShaper):
    """
    Reward is the percentage of load that is met by battery discharging.

     Return a value in [-1, 1]. Value of -1 implies that all load was loss load. Value of 1 implies all load
     was met by battery.

    """
    yaml_tag = u"!BatteryDischargeShaper"

    def __call__(self, step_info, cost_info):
        battery_discharge = self.sum_module_val(step_info, 'battery', 'provided_energy')
        load = self.sum_module_val(step_info, 'load', 'absorbed_energy')
        loss_load = self.sum_module_val(step_info, 'unbalanced_energy', 'provided_energy')

        # battery_discharge is in [0, load-loss_load]; loss_load is in [0, load].
        percent_battery = (battery_discharge - loss_load) / load
        assert -1 <= percent_battery <= 1
        return percent_battery

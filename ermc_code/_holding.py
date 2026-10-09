from dataclasses import dataclass
import numpy as np
from ermc._attainability import RefinedWrapper, Settings as RefinedSettings

@dataclass
class Settings(RefinedSettings):
    matched_controls: bool = True

class ControlledWrapper(RefinedWrapper):

    def option_ends(self, active, hit, timeout, progress, done):
        if not self.cfg.matched_controls:
            return super().option_ends(active, hit, timeout, progress, done)
        active['any_hit'] = active.get('any_hit', False) or bool(hit)
        return timeout or progress or done

    def option_hit(self, active, hit):
        return active.get('any_hit', hit) if self.cfg.matched_controls else hit

    def reset(self):
        self.u0_step_count = self.auxiliary_step_count = 0
        return super().reset()

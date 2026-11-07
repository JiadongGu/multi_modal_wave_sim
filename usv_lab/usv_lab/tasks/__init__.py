"""Task registration. Imported by Isaac Lab through the `isaaclab.tasks` entry point."""

import gymnasium as gym

gym.register(
    id="Usv-Float-Direct-v0",
    entry_point=f"{__name__}.usv_env:UsvEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.usv_env_cfg:UsvFloatEnvCfg"},
)

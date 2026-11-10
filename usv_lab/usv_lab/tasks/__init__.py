"""Task registration. Imported by Isaac Lab through the `isaaclab.tasks` entry point."""

import gymnasium as gym

gym.register(
    id="Usv-Float-Direct-v0",
    entry_point=f"{__name__}.usv_env:UsvEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.usv_env_cfg:UsvFloatEnvCfg"},
)

gym.register(
    id="Usv-StationKeep-Direct-v0",
    entry_point=f"{__name__}.station_env:UsvStationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.station_env_cfg:UsvStationEnvCfg",
        "skrl_cfg_entry_point": f"{__name__}.agents:skrl_station_ppo_cfg.yaml",
    },
)

# Same task for watching a trained policy: 16 boats with their water drawn.
gym.register(
    id="Usv-StationKeep-Play-Direct-v0",
    entry_point=f"{__name__}.station_env:UsvStationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.station_env_cfg:UsvStationPlayEnvCfg",
        "skrl_cfg_entry_point": f"{__name__}.agents:skrl_station_ppo_cfg.yaml",
    },
)

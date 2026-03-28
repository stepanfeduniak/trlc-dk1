"""Motor controller configurations for DK1 joints (draccus ChoiceRegistry)."""

import abc
from dataclasses import dataclass, field

import draccus


@dataclass(kw_only=True)
class DK1ControllerConfig(draccus.ChoiceRegistry, abc.ABC):
    """Motor controller configuration for DK1 joints."""

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@DK1ControllerConfig.register_subclass("pos_vel")
@dataclass
class PosVelControllerConfig(DK1ControllerConfig):
    """Position-Velocity controller with per-joint PI position loop."""

    @dataclass
    class JointParams:
        """Per-joint parameters for PosVel control mode (acc/dec/kp/ki)."""
        acc: float = 10.0
        dec: float = -10.0
        kp: float = 200.0
        ki: float = 10.0

    # DM4340 shoulder
    joint_1: JointParams = field(default_factory=JointParams)
    joint_2: JointParams = field(default_factory=JointParams)
    joint_3: JointParams = field(default_factory=JointParams)
    # DM4310 wrist (conservative defaults)
    joint_4: JointParams = field(
        default_factory=lambda: PosVelControllerConfig.JointParams(acc=5.0, dec=-5.0, kp=100.0, ki=5.0)
    )
    joint_5: JointParams = field(
        default_factory=lambda: PosVelControllerConfig.JointParams(acc=5.0, dec=-5.0, kp=100.0, ki=5.0)
    )
    joint_6: JointParams = field(
        default_factory=lambda: PosVelControllerConfig.JointParams(acc=5.0, dec=-5.0, kp=100.0, ki=5.0)
    )


@DK1ControllerConfig.register_subclass("torque_pos")
@dataclass
class TorquePosControllerConfig(DK1ControllerConfig):
    """Force-Position mixed controller (gripper)."""

    kp: float = 100.0  # Position P-gain (KP_APR)


@DK1ControllerConfig.register_subclass("mit")
@dataclass
class MITControllerConfig(DK1ControllerConfig):
    """MIT impedance controller with per-joint (kp, kd) gains."""

    @dataclass
    class JointParams:
        """Per-joint parameters for MIT impedance control mode (kp/kd)."""
        kp: float = 50.0   # Position stiffness (0-500)
        kd: float = 3.0     # Velocity damping (0-5)

    # DM4340 shoulder joints
    joint_1: JointParams = field(default_factory=lambda: MITControllerConfig.JointParams(kp=80.0, kd=5.0))
    joint_2: JointParams = field(default_factory=lambda: MITControllerConfig.JointParams(kp=80.0, kd=5.0))
    joint_3: JointParams = field(default_factory=lambda: MITControllerConfig.JointParams(kp=80.0, kd=5.0))
    # DM4310 wrist joints
    joint_4: JointParams = field(default_factory=lambda: MITControllerConfig.JointParams(kp=40.0, kd=1.5))
    joint_5: JointParams = field(default_factory=lambda: MITControllerConfig.JointParams(kp=10.0, kd=1.5))
    joint_6: JointParams = field(default_factory=lambda: MITControllerConfig.JointParams(kp=10.0, kd=1.5))


@DK1ControllerConfig.register_subclass("mit_a")
@dataclass
class MITAdaptiveControllerConfig(DK1ControllerConfig):
    """Adaptive MIT impedance controller: kp/kd scale with command freshness.

    Gains scale down when the inter-command interval is large (packet loss),
    making the robot compliant and preventing jerky jumps. When commands
    resume at normal rate, full stiffness is restored.

    Effective gains: kp_eff = kp * scale,  kd_eff = kd * scale
    where scale = min_scale + (1 - min_scale) * exp(-cmd_dt / expected_dt_s)
    """

    @dataclass
    class JointParams:
        """Per-joint MIT gains (kp/kd) — base values at full stiffness."""
        kp: float = 50.0
        kd: float = 3.0

    # Base MIT gains (identical defaults to MITControllerConfig)
    joint_1: JointParams = field(default_factory=lambda: MITAdaptiveControllerConfig.JointParams(kp=80.0, kd=5.0))
    joint_2: JointParams = field(default_factory=lambda: MITAdaptiveControllerConfig.JointParams(kp=80.0, kd=5.0))
    joint_3: JointParams = field(default_factory=lambda: MITAdaptiveControllerConfig.JointParams(kp=80.0, kd=5.0))
    joint_4: JointParams = field(default_factory=lambda: MITAdaptiveControllerConfig.JointParams(kp=40.0, kd=1.5))
    joint_5: JointParams = field(default_factory=lambda: MITAdaptiveControllerConfig.JointParams(kp=10.0, kd=1.5))
    joint_6: JointParams = field(default_factory=lambda: MITAdaptiveControllerConfig.JointParams(kp=10.0, kd=1.5))

    # Adaptive parameters
    min_scale: float = 0.15            # minimum gain multiplier (15% of base)
    expected_dt_s: float = 0.04        # expected command interval (~25Hz)

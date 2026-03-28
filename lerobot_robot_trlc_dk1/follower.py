#   Copyright 2025 The Robot Learning Company UG (haftungsbeschränkt). All rights reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from dataclasses import dataclass, field
from functools import cached_property
import math
import serial
import threading
import time
import logging
from typing import Any

from lerobot.cameras import CameraConfig
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots import Robot, RobotConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from lerobot_robot_trlc_dk1.controller_configs import (
    DK1ControllerConfig,
    PosVelControllerConfig,
    TorquePosControllerConfig,
    MITControllerConfig,
    MITHControllerConfig,
)
from lerobot_robot_trlc_dk1.motors.DM_Control_Python.DM_CAN import *

logger = logging.getLogger(__name__)


def map_range(x: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def _precise_sleep(duration: float) -> None:
    """Sleep with busy-wait tail for sub-ms accuracy."""
    if duration <= 0:
        return
    end = time.perf_counter() + duration
    remaining = duration - 0.001
    if remaining > 0:
        time.sleep(remaining)
    while time.perf_counter() < end:
        pass


@RobotConfig.register_subclass("dk1_follower")
@dataclass
class DK1FollowerConfig(RobotConfig):
    port: str
    disable_torque_on_disconnect: bool = False
    joint_velocity_scaling: float = 0.2
    max_gripper_torque: float = 1.0 # Nm (/0.00875m spur gear radius = 114N gripper force)
    joint_controller: DK1ControllerConfig = field(default_factory=PosVelControllerConfig)
    gripper_controller: DK1ControllerConfig = field(default_factory=TorquePosControllerConfig)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)


class DK1Follower(Robot):
    """
    TRLC-DK1 Follower Arm designed by The Robot Learning Company.
    """

    config_class = DK1FollowerConfig
    name = "dk1_follower"

    def __init__(self, config: DK1FollowerConfig):
        super().__init__(config)
        
        # Constants for EMIT control
        self.DM4310_TORQUE_CONSTANT = 0.945  # Nm/A
        self.EMIT_VELOCITY_SCALE = 100  # rad/s
        self.EMIT_CURRENT_SCALE = 1000  # A
        
        self.JOINT_LIMITS = {
            "joint_4": (-100/180*np.pi, 100/180*np.pi),
            "joint_5": (-90/180*np.pi, 90/180*np.pi),
        }
        
        self.DM4310_SPEED = 200/60*2*np.pi   # rad/s (200  rpm | 20.94 rad/s)
        self.DM4340_SPEED = 52.5/60*2*np.pi  # rad/s (52.5 rpm | 5.49  rad/s)

        self.config = config
        self.motors = {
            "joint_1": Motor(DM_Motor_Type.DM4340, 0x01, 0x11),
            "joint_2": Motor(DM_Motor_Type.DM4340, 0x02, 0x12),
            "joint_3": Motor(DM_Motor_Type.DM4340, 0x03, 0x13),
            "joint_4": Motor(DM_Motor_Type.DM4310, 0x04, 0x14),
            "joint_5": Motor(DM_Motor_Type.DM4310, 0x05, 0x15),
            "joint_6": Motor(DM_Motor_Type.DM4310, 0x06, 0x16),
            "gripper": Motor(DM_Motor_Type.DM4310, 0x07, 0x17),
        }
        self.control = None
        self.serial_device = None
        self.bus_connected = False

        self.gripper_open_pos = 0.0
        self.gripper_closed_pos = -4.7

        self.cameras = make_cameras_from_configs(config.cameras)

        # Servo loop state (active only for mit_h controller)
        self._servo_lock = threading.Lock()
        self._target_positions: dict[str, float] | None = None
        self._current_positions: dict[str, float] | None = None
        self._servo_thread: threading.Thread | None = None
        self._servo_running = False
        self._last_command_time: float = 0.0
        self._alpha: float = 0.02  # current smoothing factor, set by send_action()

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.bus_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.serial_device = serial.Serial(
            self.config.port, 921600, timeout=0.5)
        time.sleep(0.5)

        self.control = MotorControl(self.serial_device)
        self.bus_connected = True
        self.configure()

        for cam in self.cameras.values():
            cam.connect()

        if self._is_mit_h:
            self._servo_running = True
            self._servo_thread = threading.Thread(
                target=self._servo_loop, daemon=True, name="dk1-servo",
            )
            self._servo_thread.start()
            logger.info("Servo loop started at %.0f Hz", self.config.joint_controller.servo_rate_hz)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    @property
    def _is_mit_h(self) -> bool:
        return isinstance(self.config.joint_controller, MITHControllerConfig)

    def configure(self) -> None:

        jc = self.config.joint_controller
        is_mit = isinstance(jc, (MITControllerConfig, MITHControllerConfig))
        arm_control_mode = Control_Type.MIT if is_mit else Control_Type.POS_VEL

        for key, motor in self.motors.items():
            self.control.addMotor(motor)

            for _ in range(3):
                self.control.refresh_motor_status(motor)
                time.sleep(0.01)

            if self.control.read_motor_param(motor, DM_variable.CTRL_MODE) is not None:
                print(f"{key} ({motor.MotorType.name}) is connected.")

                if key == "gripper":
                    self.control.switchControlMode(motor, Control_Type.POS_VEL)
                else:
                    self.control.switchControlMode(motor, arm_control_mode)
                self.control.enable(motor)
            else:
                raise Exception(
                    f"Unable to read from {key} ({motor.MotorType.name}).")

        # Configure arm joint registers (PosVel only — MIT sends gains per-frame)
        if isinstance(jc, PosVelControllerConfig):
            for name in ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]:
                jp = getattr(jc, name)
                self.control.change_motor_param(self.motors[name], DM_variable.ACC, jp.acc)
                self.control.change_motor_param(self.motors[name], DM_variable.DEC, jp.dec)
                self.control.change_motor_param(self.motors[name], DM_variable.KP_APR, jp.kp)
                self.control.change_motor_param(self.motors[name], DM_variable.KI_APR, jp.ki)

        # Gripper: only kp (Torque_Pos mode) — from separate gripper_controller
        gc = self.config.gripper_controller
        if isinstance(gc, TorquePosControllerConfig):
            self.control.change_motor_param(
                self.motors["gripper"], DM_variable.KP_APR, gc.kp)

        # Open gripper and set zero position
        self.control.switchControlMode(
            self.motors["gripper"], Control_Type.VEL)
        self.control.control_Vel(self.motors["gripper"], 10.0)
        while True:
            self.control.refresh_motor_status(self.motors["gripper"])
            tau = self.motors["gripper"].getTorque()
            if tau > 1.2:
                self.control.control_Vel(self.motors["gripper"], 0.0)
                self.control.disable(self.motors["gripper"])
                self.control.set_zero_position(self.motors["gripper"])
                time.sleep(0.2)
                self.control.enable(self.motors["gripper"])
                break
            time.sleep(0.01)
        self.control.switchControlMode(
            self.motors["gripper"], Control_Type.Torque_Pos)

    def _servo_loop(self) -> None:
        """Background loop: interpolate toward target and send to motors at fixed rate.

        Alpha is set by send_action() based on the interval between commands:
        short interval (normal flow) → high alpha. Long interval (gap) → low alpha.
        The servo loop reads self._alpha each cycle and interpolates accordingly.
        """
        jc: MITHControllerConfig = self.config.joint_controller
        period = 1.0 / jc.servo_rate_hz
        alpha_grip = jc.gripper_smoothing_factor

        # Diagnostics
        cycle_count = 0
        log_interval = int(jc.servo_rate_hz)  # log every ~1s
        window_start = time.perf_counter()

        while self._servo_running:
            t0 = time.perf_counter()

            try:
                with self._servo_lock:
                    if self._target_positions is None or self._current_positions is None:
                        pass  # No target yet — fall through to rate-limit sleep
                    else:
                        alpha = self._alpha

                        for key in self._current_positions:
                            a = alpha_grip if key == "gripper" else alpha
                            self._current_positions[key] += a * (
                                self._target_positions[key] - self._current_positions[key]
                            )

                        self._send_to_motors(self._current_positions)

                        cycle_count += 1
                        if cycle_count >= log_interval:
                            now = time.perf_counter()
                            hz = cycle_count / max(1e-9, now - window_start)
                            logger.debug("servo loop: %.0f Hz  alpha=%.3f", hz, alpha)
                            cycle_count = 0
                            window_start = now
            except Exception:
                logger.error("servo loop error", exc_info=True)

            # Rate-limit: busy-wait for the remainder of the period
            elapsed = time.perf_counter() - t0
            remaining = period - elapsed
            if remaining > 0:
                _precise_sleep(remaining)

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()

        obs_dict = {}
        # Synchronize with servo thread when it owns the serial bus
        with self._servo_lock:
            for key, motor in self.motors.items():
                self.control.refresh_motor_status(motor)
                if key == "gripper":
                    obs_dict[f"{key}.pos"] = map_range(
                        motor.getPosition(), self.gripper_open_pos, self.gripper_closed_pos, 0.0, 1.0)
                else:
                    obs_dict[f"{key}.pos"] = motor.getPosition()

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images outside the lock (no serial contention)
        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read()

        return obs_dict

    def _send_to_motors(self, goal_pos: dict[str, float]) -> None:
        """Write goal positions to all motors. Must be called with _servo_lock held (or from single thread)."""
        for key, motor in self.motors.items():
            if key == "gripper":
                self.control.refresh_motor_status(motor)
                gripper_goal_pos_mapped = map_range(
                    goal_pos[key], 0.0, 1.0, self.gripper_open_pos, self.gripper_closed_pos,
                )
                self.control.control_pos_force(
                    motor, gripper_goal_pos_mapped,
                    self.DM4310_SPEED * self.EMIT_VELOCITY_SCALE,
                    i_des=self.config.max_gripper_torque / self.DM4310_TORQUE_CONSTANT * self.EMIT_CURRENT_SCALE,
                )
            else:
                pos = goal_pos[key]
                if key in self.JOINT_LIMITS:
                    pos = np.clip(pos, self.JOINT_LIMITS[key][0], self.JOINT_LIMITS[key][1])

                jc = self.config.joint_controller
                if isinstance(jc, (MITControllerConfig, MITHControllerConfig)):
                    jp = getattr(jc, key)
                    kp = jp["kp"] if isinstance(jp, dict) else jp.kp
                    kd = jp["kd"] if isinstance(jp, dict) else jp.kd
                    self.control.controlMIT(motor, kp, kd, pos, dq=0.0, tau=0.0)
                else:
                    max_speed = (
                        self.DM4310_SPEED
                        if motor.MotorType == DM_Motor_Type.DM4310
                        else self.DM4340_SPEED
                    )
                    self.control.control_Pos_Vel(
                        motor, pos, self.config.joint_velocity_scaling * max_speed,
                    )

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        if self._is_mit_h:
            # Non-blocking: update target for servo thread
            now = time.perf_counter()
            with self._servo_lock:
                if self._current_positions is None:
                    self._current_positions = dict(goal_pos)
                self._target_positions = goal_pos

                # Adaptive alpha from inter-command interval
                jc: MITHControllerConfig = self.config.joint_controller
                if self._last_command_time > 0:
                    cmd_dt = now - self._last_command_time
                    # exp(-dt/expected_dt): 1.0 at expected rate, decays for longer gaps
                    t = math.exp(-cmd_dt / jc.expected_dt_s)
                    self._alpha = jc.smoothing_min + (jc.smoothing_max - jc.smoothing_min) * t
                else:
                    self._alpha = jc.smoothing_max
                self._last_command_time = now
        else:
            self._send_to_motors(goal_pos)

        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Stop servo thread before touching serial
        self._servo_running = False
        if self._servo_thread is not None:
            self._servo_thread.join(timeout=2.0)
            self._servo_thread = None

        if self.config.disable_torque_on_disconnect:
            for motor in self.motors.values():
                self.control.disable(motor)
        else:
            self.control.serial_.close()
        self.bus_connected = False

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")

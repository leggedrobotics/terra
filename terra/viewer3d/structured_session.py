"""Manual structured actions on the saved pull-rule states."""

import math

import numpy as np

from .pull_session import ACTION_NAMES, PullSession


_LEGACY_PRECISION_SETTINGS = dict(
    terminate_unloaded_dead_ends=False, precision_safety=False, precision_edge_weight=-1.,
)


def _protocol_with_precision_defaults(protocol):
    if protocol is None:
        return None
    return dict(protocol, episode={**_LEGACY_PRECISION_SETTINGS, **protocol["episode"]})


def load_structured_protocol(path):
    """Read explicit episode settings from a trainer config or a flat JSON file."""
    import json
    from pathlib import Path
    from terra.structured_actions import StructuredTimeConfig
    from terra.structured_episode import StructuredEpisodeConfig

    values = json.loads(Path(path).read_text())
    if not isinstance(values, dict):
        raise ValueError("Structured protocol must be a JSON object.")
    values = values.get("training", values)
    if not isinstance(values, dict):
        raise ValueError("Structured protocol training settings must be an object.")
    timing = values.get("timing")
    if not isinstance(timing, dict):
        raise ValueError("Structured protocol requires an explicit timing object.")
    if set(timing) != set(StructuredTimeConfig._fields):
        raise ValueError("Structured protocol timing must specify every StructuredTimeConfig field.")
    for name, value in timing.items():
        if name == "setup_per_dig":
            if not isinstance(value, bool):
                raise ValueError("timing.setup_per_dig must be boolean.")
        elif (isinstance(value, bool) or not isinstance(value, (int, float))
              or not math.isfinite(value) or value < 0
              or (name == "nav_speed_mps" and value == 0)):
            raise ValueError(f"Invalid timing value: {name}.")
    required = (set(StructuredEpisodeConfig._fields)
                - {"time_limit", "terminate_loaded_dead_ends", *_LEGACY_PRECISION_SETTINGS})
    if required - values.keys():
        raise ValueError("Structured protocol is missing episode values: " + ", ".join(sorted(required - values.keys())))
    episode = {name: values[name] for name in required}
    if "time_limit" not in values and "no_time_limit" not in values:
        raise ValueError("Structured protocol requires time_limit or no_time_limit.")
    for name in ("time_limit", "no_time_limit", "terminate_loaded_dead_ends",
                 "terminate_unloaded_dead_ends", "precision_safety"):
        if name in values and not isinstance(values[name], bool):
            raise ValueError(f"{name} must be boolean.")
    if ("time_limit" in values and "no_time_limit" in values
            and values["time_limit"] == values["no_time_limit"]):
        raise ValueError("Conflicting time_limit and no_time_limit values.")
    episode["time_limit"] = values.get("time_limit", not values.get("no_time_limit", False))
    # Old trainer configs predate this opt-in behavior. Never reinterpret them.
    episode["terminate_loaded_dead_ends"] = values.get("terminate_loaded_dead_ends", False)
    episode.update({name: values.get(name, default) for name, default in _LEGACY_PRECISION_SETTINGS.items()})
    weight = episode["precision_edge_weight"]
    if (isinstance(weight, bool) or not isinstance(weight, (int, float))
            or not math.isfinite(weight) or not (weight == -1. or 0. <= weight <= 1.)):
        raise ValueError("precision_edge_weight must be -1 (legacy) or between 0 and 1.")
    for name in required:
        value = episode[name]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise ValueError(f"Invalid episode value: {name}.")
        if name in ("time_budget_s", "decision_limit") and value <= 0:
            raise ValueError(f"{name} must be positive.")
        if name == "decision_limit" and not isinstance(value, int):
            raise ValueError("decision_limit must be an integer.")
    return StructuredTimeConfig(**timing), StructuredEpisodeConfig(**episode)


class StructuredPullSession(PullSession):
    """Opt-in manual controls sharing the native structured training transition."""

    action_mode = "structured_v1"

    def __init__(self, *, time_budget_s=14400.0, decision_budget=450, resume_recording=None,
                 structured_protocol=None, **kwargs):
        import jax
        import jax.numpy as jnp
        from terra.structured_actions import (
            StructuredClock, StructuredTimeConfig, structured_action_masks, structured_transition,
            structured_loaded_dead_end,
        )
        from terra.structured_episode import (
            StructuredEpisodeConfig, structured_episode_limits, structured_episode_termination,
        )

        if isinstance(time_budget_s, bool) or not math.isfinite(time_budget_s) or time_budget_s <= 0:
            raise ValueError("time_budget_s must be finite and positive.")
        if isinstance(decision_budget, bool) or not isinstance(decision_budget, int) or decision_budget <= 0:
            raise ValueError("decision_budget must be a positive integer.")
        self.time_budget_s = float(time_budget_s)
        self.decision_budget = decision_budget
        self.timing = StructuredTimeConfig()
        self.episode_config = StructuredEpisodeConfig(
            time_budget_s=self.time_budget_s, decision_limit=decision_budget)
        if structured_protocol is not None:
            self.timing, self.episode_config = load_structured_protocol(structured_protocol)
        self.structured_protocol = dict(episode=self.episode_config._asdict(), timing=self.timing._asdict())
        self._structured_ready = False
        warmup = kwargs.pop("warmup", True)
        super().__init__(warmup=False, **kwargs)
        config = jax.tree_util.tree_map(
            lambda x: np.asarray(x).item() if np.ndim(x) == 0 else x,
            self.state.env_cfg._replace(precision_safety=self.episode_config.precision_safety))

        def native(state):
            return state._replace(env_cfg=config._replace(
                enforce_foundation_border_alignment=state.env_cfg.enforce_foundation_border_alignment))

        @jax.jit
        def advance(state, action, clock):
            budget, _ = structured_episode_limits(native(state), timing=self.timing, config=self.episode_config)
            result = structured_transition(native(state), action, clock=clock,
                                           timing=self.timing, time_budget_s=budget,
                                           precision_edge_weight=self.episode_config.precision_edge_weight)
            return result._replace(state=result.state._replace(env_cfg=state.env_cfg))

        self._structured_advance = advance
        self._structured_masks = jax.jit(lambda state: structured_action_masks(native(state)))
        self._structured_termination = jax.jit(lambda state, elapsed, do_mask: structured_episode_termination(
            native(state), elapsed, timing=self.timing, config=self.episode_config,
            dead_end=structured_loaded_dead_end(native(state), do_mask),
            unloaded_dead_end=(native(state)._unloaded_dead_end()
                               if self.episode_config.terminate_unloaded_dead_ends else jnp.bool_(False))))
        self._structured_limits = jax.jit(lambda state: structured_episode_limits(
            native(state), timing=self.timing, config=self.episode_config))
        self._initial_clock = StructuredClock()
        self._structured_ready = True
        self.reset()
        if warmup:
            from terra.structured_actions import StructuredAction
            jax.block_until_ready(self._structured_advance(
                self.state, StructuredAction(jnp.int32(7), jnp.int32(1), jnp.int32(-1)),
                self.clocks[-1]))
        if resume_recording is not None:
            self.restore_recording(resume_recording)

    def _with_protocol_rules(self, state):
        import jax.numpy as jnp
        return state._replace(env_cfg=state.env_cfg._replace(
            precision_safety=jnp.asarray(self.episode_config.precision_safety, dtype=jnp.bool_)))

    def reset(self, **kwargs):
        if not self._structured_ready:
            return super().reset(**kwargs)
        selected = self.selected_case
        case_id = kwargs.get("case_id")
        case_id = selected["case_id"] if case_id is None else str(case_id)
        precision = kwargs.get("precision")
        precision = selected["precision"] if precision is None else precision
        start = kwargs.get("start")
        start = selected["start"] if start is None else start
        if not isinstance(precision, bool) or isinstance(start, bool) or not isinstance(start, int):
            raise ValueError("precision must be boolean; start must be an integer.")
        if (case_id, precision, start) not in self.initials:
            raise ValueError("Choose a listed saved case, precision mode, and start.")
        if set(kwargs) - {"case_id", "precision", "start"}:
            raise ValueError("Reset accepts case_id, precision, and start.")
        # Parent reset exports the previous nonempty recording before resetting.
        previous = self.__dict__.copy()
        try:
            if self.recorder is not None and self.actions:
                self.export()
                self.actions = []
            self.elapsed_times = [0.0]
            self.clocks = [self._initial_clock]
            key = (case_id, precision, start)
            episode, initial = self.initials[key]
            # Parent reset derives its diagnostic frame and export state from
            # this bank entry; both must carry the same rules as the transition.
            self.initials = {**self.initials, key: (episode, self._with_protocol_rules(initial))}
            budget, limit = self._structured_limits(self.initials[key][1])
            self.time_budget_s, self.decision_budget = float(budget), int(limit)
            super().reset(**kwargs)
        except Exception:
            self.__dict__.clear()
            self.__dict__.update(previous)
            raise
        self.recorder.metadata.update(
            action_mode=self.action_mode, time_budget_s=self.time_budget_s,
            decision_budget=self.decision_budget, timing_model="material_time_v1",
            structured_protocol=self.structured_protocol,
            heading_convention="cabin index relative to chassis; 0..11 in 30 degree steps",
            action_schema="action type plus amount (move/turn) or heading (DO)",
        )
        # The primitive horizon remains in the underlying saved state for legacy
        # reward compatibility; the structured episode has its own budget.
        self.recorder.metadata.pop("max_steps", None)
        return self.payload()

    def _frame(self, **kwargs):
        frame = super()._frame(**kwargs)
        if not self._structured_ready:
            return frame
        import jax
        masks = jax.device_get(self._structured_masks(self.state))
        elapsed = self.elapsed_times[-1]
        decisions = len(self.states) - 1
        physical_success = bool(frame["task_done"])
        terminal = jax.device_get(self._structured_termination(self.state, elapsed, masks["do_mask"]))
        success = bool(terminal.task_done)
        loaded = bool(self.state._get_current_agent_state().loaded[0] > 0)
        loaded_dead_end = bool(terminal.dead_end) and loaded
        unloaded_dead_end = bool(terminal.dead_end) and not loaded
        reason = ("success" if success else "loaded_no_unload" if loaded_dead_end
                  else "unloaded_dead_end" if unloaded_dead_end
                  else "time_budget" if self.episode_config.time_limit and elapsed >= self.time_budget_s
                  else "decision_budget" if decisions >= self.decision_budget else None)
        frame["done"] = bool(terminal.done)
        frame["task_done"] = success
        frame["step"] = decisions
        diag = frame["diagnostics"]
        diag.update(
            structured_actions={key: np.asarray(value, bool).tolist() for key, value in masks.items()},
            elapsed_time_s=elapsed, remaining_time_s=max(0.0, self.time_budget_s - elapsed),
            time_budget_s=self.time_budget_s, decisions=decisions,
            time_limit=self.episode_config.time_limit,
            decision_budget=self.decision_budget, termination_reason=reason,
            timing_model="material_time_v1", physical_task_complete=physical_success,
            terminal_reward=float(terminal.reward), loaded_dead_end=loaded_dead_end,
            unloaded_dead_end=unloaded_dead_end,
        )
        diag["structured_actions"]["current_heading"] = int(self.state._get_current_agent_state().angle_cabin[0])
        diag.pop("step_budget", None)
        diag.pop("remaining_steps", None)
        return frame

    @staticmethod
    def _action(action, amount, heading):
        from terra.structured_actions import StructuredAction
        if isinstance(action, bool) or not isinstance(action, int) or action not in range(8):
            raise ValueError("action must be an integer from 0 to 7.")
        amount = (5 if action in (0, 1) else 1) if amount is None else amount
        if isinstance(amount, bool) or not isinstance(amount, int):
            raise ValueError("amount must be an integer.")
        maximum = 5 if action in (0, 1) else 6 if action in (2, 3) else 1
        if not 1 <= amount <= maximum:
            raise ValueError(f"amount must be from 1 to {maximum} for this action.")
        heading = -1 if heading is None else heading
        if isinstance(heading, bool) or not isinstance(heading, int) or heading not in range(-1, 12):
            raise ValueError("heading must be -1 (current) or an integer from 0 to 11.")
        if action != 6 and heading != -1:
            raise ValueError("heading is only used by DO.")
        return StructuredAction(action, amount, heading)

    def step(self, action, *, amount=None, heading=None, continue_after_timeout=False):
        import jax
        import jax.numpy as jnp
        from terra.structured_actions import StructuredClock

        request = self._action(action, amount, heading)
        if not isinstance(continue_after_timeout, bool):
            raise ValueError("continue_after_timeout must be boolean.")
        previous = self.recorder.frames[-1]
        exploring = self.exploring or (continue_after_timeout and previous["done"])
        if previous["done"] and not exploring:
            raise RuntimeError("Episode ended. Reset, undo, or explicitly continue exploring.")
        masks = previous["diagnostics"]["structured_actions"]
        chosen_heading = masks["current_heading"] if request.heading == -1 else request.heading
        valid = (masks["move_mask"][action][request.amount - 1] if action < 2 else
                 masks["turn_mask"][action - 2][request.amount - 1] if action < 4 else
                 masks["do_mask"][chosen_heading] if action == 6 else True)
        if not valid:
            raise ValueError("This action has no executable outcome. Select an enabled distance, turn or work heading.")
        result = self._structured_advance(self.state, jax.tree_util.tree_map(jnp.int32, request), self.clocks[-1])
        reward, duration, info = jax.device_get((result.reward, result.duration_s, result.info))
        if (not np.isfinite(reward) or not np.isfinite(duration) or duration < 0
                or any(bool(info[name]) for name in ("target_mutation", "obstacle_mutation"))
                or int(info["transition_mass_residual"])):
            raise RuntimeError("Native transition integrity failed; the action was not committed.")
        old_exploring = self.exploring
        self.exploring = exploring
        self.states.append(result.state)
        # Match the trainer's float32 accumulation at episode boundaries.
        self.elapsed_times.append(float(np.float32(self.elapsed_times[-1]) + np.float32(duration)))
        self.clocks.append(StructuredClock(info["time_visit_open"], info["time_moved"]))
        record = dict(action=action, amount=request.amount, heading=request.heading)
        message = f"{ACTION_NAMES[action].capitalize()}: {float(duration):.1f} estimated seconds."
        try:
            frame = self._frame(reward=float(reward), action=action,
                                effect=bool(info["action_had_effect"]), message=message)
            terminal_bonus = 0.0
            if frame["done"] and not previous["done"] and not exploring:
                terminal_bonus = frame["diagnostics"]["terminal_reward"]
            frame["reward"] = float(reward) + terminal_bonus
            frame["diagnostics"].update(
                structured_action=record, duration_s=float(duration), terminal_reward=terminal_bonus,
            )
        except Exception:
            self.states.pop()
            self.elapsed_times.pop()
            self.clocks.pop()
            self.exploring = old_exploring
            raise
        self.actions.append(record)
        self.recorder.frames.append(frame)
        return frame

    def undo(self):
        if not self.actions:
            raise RuntimeError("Already at the saved initial state.")
        self.elapsed_times.pop()
        self.clocks.pop()
        super().undo()
        self._refresh_frame()
        return self.payload()

    def export(self):
        import json
        from pathlib import Path

        result = super().export()
        clock_path = Path(result["paths"]["native_states"]).with_name("structured_clock.json")
        clock_path.write_text(json.dumps(dict(
            elapsed_time_s=self.elapsed_times,
            clocks=[dict(visit_open=bool(clock.visit_open), moved=bool(clock.moved))
                    for clock in self.clocks],
            time_budget_s=self.time_budget_s, decision_budget=self.decision_budget,
            structured_protocol=self.structured_protocol,
        ), indent=2) + "\n")
        result["paths"]["structured_clock"] = str(clock_path)
        return result

    def restore_recording(self, folder):
        """Restore a trusted native export, including arguments and time history."""
        import json
        import pickle
        from pathlib import Path
        import jax
        import jax.numpy as jnp
        from terra.structured_actions import StructuredClock
        from .replay import ReplayRecorder, load_replay

        folder = Path(folder)
        with (folder / "states.pkl").open("rb") as stream:
            saved = pickle.load(stream)
        clock = json.loads((folder / "structured_clock.json").read_text())
        replay = load_replay(folder / "replay.json.gz")
        if saved["metadata"].get("action_mode") != self.action_mode:
            raise ValueError("Resume requires a structured-action native export.")
        exported_protocol = _protocol_with_precision_defaults(clock.get("structured_protocol"))
        metadata_protocol = _protocol_with_precision_defaults(saved["metadata"].get("structured_protocol"))
        replay_protocol = _protocol_with_precision_defaults(replay["metadata"].get("structured_protocol"))
        if exported_protocol != metadata_protocol or exported_protocol != replay_protocol:
            raise ValueError("Recording protocol metadata is inconsistent.")
        if exported_protocol is None:
            from terra.structured_actions import StructuredTimeConfig
            from terra.structured_episode import StructuredEpisodeConfig
            # Before loaded-dead-end handling, viewer frames had no such
            # diagnostic. Preserve that historical behavior explicitly.
            dead_end_enabled = any("loaded_dead_end" in frame.get("diagnostics", {})
                                   for frame in replay["frames"])
            legacy = dict(timing=StructuredTimeConfig()._asdict(), episode=StructuredEpisodeConfig(
                time_budget_s=clock["time_budget_s"], decision_limit=clock["decision_budget"],
                terminate_loaded_dead_ends=dead_end_enabled)._asdict())
            if self.structured_protocol != legacy:
                raise ValueError("Old recordings require their original fixed budgets, default timing and "
                                 f"terminate_loaded_dead_ends={dead_end_enabled}.")
        elif exported_protocol != self.structured_protocol:
            raise ValueError("Resume must preserve the exported structured protocol and timing.")
        selected = saved["metadata"]["selected_case"]
        key = (str(selected["case_id"]), selected["precision"], selected["start"])
        if key not in self.initials:
            raise ValueError("The exported case is not in the current saved state bank.")
        budget, limit = self._structured_limits(self.initials[key][1])
        if clock["time_budget_s"] != float(budget) or clock["decision_budget"] != int(limit):
            raise ValueError("Resume must preserve the exported time and decision budgets.")
        length = len(saved["states"])
        if (length != len(saved["actions"]) + 1 or length != len(replay["frames"])
                or length != len(clock["elapsed_time_s"]) or length != len(clock["clocks"])):
            raise ValueError("Recording state, action and clock histories have different lengths.")
        times = np.asarray(clock["elapsed_time_s"], dtype=float)
        if not np.all(np.isfinite(times)) or np.any(times < 0) or np.any(np.diff(times) < 0):
            raise ValueError("Recording elapsed time must be finite, nonnegative and monotonic.")
        self.reset(**saved["metadata"]["selected_case"])
        self.states = [self._with_protocol_rules(state)
                       for state in jax.tree_util.tree_map(jnp.asarray, saved["states"])]
        self.actions = saved["actions"]
        self.elapsed_times = times.tolist()
        self.clocks = [StructuredClock(bool(value["visit_open"]), bool(value["moved"]))
                       for value in clock["clocks"]]
        self.recorder = ReplayRecorder(metadata=replay["metadata"])
        self.recorder.metadata["structured_protocol"] = self.structured_protocol
        self.recorder.frames.extend(replay["frames"])
        self.exploring = bool(replay["frames"][-1]["diagnostics"]["exploring"])
        self._refresh_frame()
        return self.payload()

    def _refresh_frame(self):
        """Re-evaluate availability after code updates without replaying actions."""
        old = self.recorder.frames[-1]
        diagnostics = old["diagnostics"]
        frame = self._frame(reward=old["reward"], action=old["action"],
                            effect=diagnostics["action_had_effect"], message=diagnostics["message"])
        for key in ("structured_action", "duration_s", "terminal_reward"):
            if key in diagnostics:
                frame["diagnostics"][key] = diagnostics[key]
        self.recorder.frames[-1] = frame

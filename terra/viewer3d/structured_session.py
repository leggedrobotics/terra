"""Manual structured actions on the saved pull-rule states."""

import math

import numpy as np

from .pull_session import ACTION_NAMES, PullSession


class StructuredPullSession(PullSession):
    """Opt-in manual controls sharing the native structured training transition."""

    action_mode = "structured_v1"

    def __init__(self, *, time_budget_s=14400.0, decision_budget=450, resume_recording=None, **kwargs):
        import jax
        import jax.numpy as jnp
        from terra.structured_actions import (
            StructuredClock, structured_action_masks, structured_transition, structured_termination,
        )

        if isinstance(time_budget_s, bool) or not math.isfinite(time_budget_s) or time_budget_s <= 0:
            raise ValueError("time_budget_s must be finite and positive.")
        if isinstance(decision_budget, bool) or not isinstance(decision_budget, int) or decision_budget <= 0:
            raise ValueError("decision_budget must be a positive integer.")
        self.time_budget_s = float(time_budget_s)
        self.decision_budget = decision_budget
        self._structured_ready = False
        warmup = kwargs.pop("warmup", True)
        super().__init__(warmup=False, **kwargs)
        config = jax.tree_util.tree_map(
            lambda x: np.asarray(x).item() if np.ndim(x) == 0 else x, self.state.env_cfg)

        def native(state):
            return state._replace(env_cfg=config._replace(
                enforce_foundation_border_alignment=state.env_cfg.enforce_foundation_border_alignment))

        @jax.jit
        def advance(state, action, clock):
            result = structured_transition(native(state), action, clock=clock,
                                           time_budget_s=self.time_budget_s)
            return result._replace(state=result.state._replace(env_cfg=state.env_cfg))

        self._structured_advance = advance
        self._structured_masks = jax.jit(lambda state: structured_action_masks(native(state)))
        self._structured_termination = jax.jit(lambda state, elapsed: structured_termination(
            native(state), elapsed, time_budget_s=self.time_budget_s, decision_limit=self.decision_budget))
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
            super().reset(**kwargs)
        except Exception:
            self.__dict__.clear()
            self.__dict__.update(previous)
            raise
        self.recorder.metadata.update(
            action_mode=self.action_mode, time_budget_s=self.time_budget_s,
            decision_budget=self.decision_budget, timing_model="material_time_v1",
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
        terminal = jax.device_get(self._structured_termination(self.state, elapsed))
        success = bool(terminal.task_done)
        reason = ("success" if success else "time_budget" if elapsed >= self.time_budget_s
                  else "decision_budget" if decisions >= self.decision_budget else None)
        frame["done"] = bool(terminal.done)
        frame["task_done"] = success
        frame["step"] = decisions
        diag = frame["diagnostics"]
        diag.update(
            structured_actions={key: np.asarray(value, bool).tolist() for key, value in masks.items()},
            elapsed_time_s=elapsed, remaining_time_s=max(0.0, self.time_budget_s - elapsed),
            time_budget_s=self.time_budget_s, decisions=decisions,
            decision_budget=self.decision_budget, termination_reason=reason,
            timing_model="material_time_v1", physical_task_complete=physical_success,
            terminal_reward=float(terminal.reward),
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
        self.elapsed_times.append(self.elapsed_times[-1] + float(duration))
        self.clocks.append(StructuredClock(info["time_visit_open"], info["time_moved"]))
        record = dict(action=action, amount=request.amount, heading=request.heading)
        message = f"{ACTION_NAMES[action].capitalize()}: {float(duration):.1f} estimated seconds."
        try:
            frame = self._frame(reward=float(reward), action=action,
                                effect=bool(info["action_had_effect"]), message=message)
            terminal_bonus = 0.0
            if frame["done"] and not previous["done"]:
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
        if clock["time_budget_s"] != self.time_budget_s or clock["decision_budget"] != self.decision_budget:
            raise ValueError("Resume must preserve the exported time and decision budgets.")
        length = len(saved["states"])
        if (length != len(saved["actions"]) + 1 or length != len(replay["frames"])
                or length != len(clock["elapsed_time_s"]) or length != len(clock["clocks"])):
            raise ValueError("Recording state, action and clock histories have different lengths.")
        times = np.asarray(clock["elapsed_time_s"], dtype=float)
        if not np.all(np.isfinite(times)) or np.any(times < 0) or np.any(np.diff(times) < 0):
            raise ValueError("Recording elapsed time must be finite, nonnegative and monotonic.")
        self.reset(**saved["metadata"]["selected_case"])
        self.states = jax.tree_util.tree_map(jnp.asarray, saved["states"])
        self.actions = saved["actions"]
        self.elapsed_times = times.tolist()
        self.clocks = [StructuredClock(bool(value["visit_open"]), bool(value["moved"]))
                       for value in clock["clocks"]]
        self.recorder = ReplayRecorder(metadata=replay["metadata"])
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

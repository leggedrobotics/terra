"""Manual inspection of the saved October 8 pull-rule episodes."""

from datetime import datetime
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np

from .replay import ReplayRecorder
from .snapshots import snapshot_from_timestep


DEFAULT_OUTPUT = Path("manual-recordings")
ACTION_NAMES = ("forward", "backward", "base clockwise", "base counterclockwise",
                "cabin clockwise", "cabin counterclockwise", "DO", "wait")


class PullSession:
    """One simulated excavator; native actions, authoritative state undo."""

    def __init__(self, *, initial_states, output=DEFAULT_OUTPUT,
                 case_id="17411", precision=True, start=0, warmup=True):
        import jax
        import jax.numpy as jnp

        from terra.actions import TrackedAction
        from terra.env import TerraEnv
        from terra.wrappers import TraversabilityMaskWrapper

        self.initial_path = Path(initial_states).resolve()
        self.output = Path(output).resolve()
        with self.initial_path.open("rb") as stream:
            saved = pickle.load(stream)
        self.initials = {}
        cases = {}
        for episode, state in zip(saved["episodes"], saved["initial"]):
            if episode["decoder"] != "greedy":
                continue
            key = (str(episode["source_slot"]), episode["mode"] == "precision", episode["start"])
            if key in self.initials:
                raise ValueError(f"Duplicate saved initial state {key}")
            if not bool(state.env_cfg.pull_direction_alignment) or int(state.env_cfg.max_steps_in_episode) != 450:
                raise ValueError("The inspector requires saved pull-rule states with a 450-action horizon.")
            if int(state.agent.num_agents) != 1 or int(state.agent.agent_states[0].agent_type[0]) != 0:
                raise ValueError("The inspector supports one tracked excavator.")
            self.initials[key] = (episode, jax.tree_util.tree_map(jnp.asarray, state))
            case = cases.setdefault(key[0], dict(id=key[0], source_slot=episode["source_slot"],
                                                title=f"{episode['source_slot']} · {episode['condition']}",
                                                modes=[], starts=[]))
            if episode["mode"] not in case["modes"]:
                case["modes"].append(episode["mode"])
            if episode["start"] not in case["starts"]:
                case["starts"].append(episode["start"])
        self.cases = list(cases.values())
        first_cfg = next(iter(self.initials.values()))[1].env_cfg
        static_cfg = jax.tree_util.tree_map(
            lambda x: np.asarray(x).item() if np.ndim(x) == 0 else x, first_cfg)
        for _, state in self.initials.values():
            for field in first_cfg._fields:
                if field == "enforce_foundation_border_alignment":
                    continue
                for left, right in zip(jax.tree_util.tree_leaves(getattr(first_cfg, field)),
                                       jax.tree_util.tree_leaves(getattr(state.env_cfg, field))):
                    np.testing.assert_array_equal(left, right, err_msg=f"saved rule mismatch: {field}")

        def native(state):
            # Capture scalar settings so unrelated reward/agent branches fold away.
            return state._replace(env_cfg=static_cfg._replace(
                enforce_foundation_border_alignment=state.env_cfg.enforce_foundation_border_alignment))

        @jax.jit
        def advance(state, action):
            old = native(state)
            wrapped_action = TrackedAction.new(jnp.asarray([action], jnp.int8))
            new = old._step(wrapped_action)
            info = TerraEnv._transition_diagnostics(old, new)
            new = TerraEnv._accumulate_productive_workspace_cycles(old, new, info)
            reward, _ = old._get_reward(new, wrapped_action)
            changed = jnp.any(old.world.action_map.map != new.world.action_map.map)
            new = TraversabilityMaskWrapper.wrap(new, update_reachability=(action == 6) & changed)
            done, task_done = new._is_done(new.world.action_map.map, new.world.target_map.map)
            # Keep input leaf dtypes stable across actions and case changes.
            new = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.asarray(x).dtype),
                                         new._replace(env_cfg=state.env_cfg))
            return new, reward, done, task_done, info

        @jax.jit
        def probe(state):
            state = native(state)
            cur = state._get_current_agent_state()
            shape = state.world.target_map.map.shape
            target = state.world.target_map.map
            terrain = state.world.action_map.map
            remaining = (target < 0) & (terrain > target)
            context = state._fresh_trench_pose_valid_cells()
            footprint = state._active_base_footprint_mask()

            def heading(offset):
                rotated = state._set_current_agent_state(cur._replace(
                    angle_cabin=((cur.angle_cabin + offset) % static_cfg.agent.angles_cabin).astype(cur.angle_cabin.dtype)))
                cone = rotated._build_dig_dump_cone()
                mask, volume, relift, admitted = rotated._dig_eligibility(
                    cone, fresh_trench_context=context, outside_base_footprint=~footprint.reshape(-1))
                fresh = mask.reshape(shape) & remaining & admitted & ~relift & (cur.loaded[0] == 0)
                obstacle = jnp.any(cone.reshape(shape) & (state.world.padding_mask.map == 1))
                return fresh, volume, relift, admitted, obstacle

            fresh, volume, relift, admitted, obstacle = jax.vmap(heading)(jnp.arange(12))
            done, task_done = state._is_done(terrain, target)
            completion = state._get_task_completion(terrain, target)
            loaded = cur.loaded[0]
            return dict(fresh=fresh, volume=volume, relift=relift, admitted=admitted,
                        obstacle=obstacle, remaining=remaining, footprint=footprint,
                        cone=state._build_dig_dump_cone().reshape(shape),
                        permission=context[2].reshape(shape) & remaining,
                        base_moves=state._movement_feasibility_tracked(),
                        loaded=loaded, done=done, task_done=task_done,
                        required=jnp.maximum(-target, 0).sum(),
                        dug=jnp.minimum(jnp.maximum(-terrain, 0), jnp.maximum(-target, 0)).sum(),
                        mass=terrain.astype(jnp.int32).sum() + loaded,
                        completion=completion, reach=jnp.stack(state._dig_cone_radius_bounds()))

        @jax.jit
        def dump_probe(state, offset):
            state = native(state)
            cur = state._get_current_agent_state()
            rotated = state._set_current_agent_state(cur._replace(angle_cabin=((cur.angle_cabin + offset) % 12).astype(cur.angle_cabin.dtype)))
            after = rotated._handle_dump()
            delta = after.world.action_map.map.astype(jnp.int32) - state.world.action_map.map.astype(jnp.int32)
            discharged = (cur.loaded[0] > 0) & (after._get_current_agent_state().loaded[0] == 0)
            accepted = discharged & (jnp.sum(jnp.where(state._accepted_dump_mask(), delta, 0)) == cur.loaded[0])
            return discharged, accepted, delta > 0

        self._advance = advance
        self._probe = probe
        self._dump_probe = dump_probe
        self._precision_band = jax.jit(lambda state: native(state)._get_precision_required_band())
        self.recorder = None
        self.reset(case_id=case_id, precision=precision, start=start)
        if warmup:
            # Compile without consuming an action or altering the saved initial state.
            warmed = self._advance(self.state, jnp.int32(7))
            jax.block_until_ready(warmed)
            jax.block_until_ready(self._advance(warmed[0], jnp.int32(7)))
            jax.block_until_ready(self._dump_probe(self.state, jnp.int32(0)))

    @property
    def state(self):
        return self.states[-1]

    def payload(self):
        return dict(mode="manual", replay=self.recorder.to_dict(), cases=self.cases,
                    selected_case=self.selected_case, can_undo=bool(self.actions), exploring=self.exploring)

    def reset(self, *, case_id=None, precision=None, start=None):
        import jax
        import jax.numpy as jnp

        selected = getattr(self, "selected_case", dict(case_id="17411", precision=True, start=0))
        case_id = selected["case_id"] if case_id is None else str(case_id)
        precision = selected["precision"] if precision is None else precision
        start = selected["start"] if start is None else start
        if not isinstance(precision, bool) or isinstance(start, bool) or not isinstance(start, int):
            raise ValueError("precision must be boolean; start must be an integer.")
        key = (case_id, precision, start)
        if key not in self.initials:
            raise ValueError("Choose a listed saved case, precision mode, and start.")
        if self.recorder is not None and self.actions:
            self.export()
        episode, state = self.initials[key]
        self.selected_case = dict(case_id=case_id, precision=precision, start=start)
        self.states, self.actions = [state], []
        self.exploring = False
        self.band = np.asarray(jax.device_get(self._precision_band(state)), bool)
        self.initial_mass = int(np.asarray(state.world.action_map.map, np.int32).sum())
        self.recorder = ReplayRecorder(metadata=dict(
            title=f"{episode['source_slot']} · {episode['condition']} · {'precision' if precision else 'bulk'}",
            source="Native pull-rule manual simulation", initial_states=str(self.initial_path),
            selected_case=self.selected_case, reset_seed=episode["reset_seed"], max_steps=450,
            source_id=episode["source_id"], height_units="abstract soil units, not physical depth",
            rules=dict(pull_direction_alignment=True, dig_pull_min_length_m=float(state.env_cfg.dig_pull_min_length_m),
                       dig_working_strip_width_m=float(state.env_cfg.dig_working_strip_width_m),
                       edge_band_width_m=float(state.env_cfg.edge_band_width_m),
                       edge_pull_tolerance_rad=float(state.env_cfg.edge_pull_tolerance_rad),
                       dig_min_radius_m=float(state.env_cfg.agent.dig_min_radius_m),
                       dump_max_radius_m=float(state.env_cfg.agent.dump_max_radius_m),
                       dug_clearance_m=float(state.env_cfg.agent.dug_clearance_m),
                       centre_chassis_on_base=bool(state.env_cfg.agent.centre_chassis_on_base)),
        ))
        self.recorder.frames.append(self._frame(reward=0.0, action=None, effect=None,
                                               message="Ready at the saved native initial pose."))
        return self.payload()

    def _frame(self, *, reward, action, effect, message):
        import jax
        import jax.numpy as jnp

        result = jax.device_get(self._probe(self.state))
        loaded = int(result["loaded"])
        dump_any = np.zeros(12, bool)
        dump_accepted = np.zeros(12, bool)
        dump_masks = np.zeros((12,) + self.band.shape, bool)
        if loaded:
            for offset in range(12):
                physical, accepted, mask = jax.device_get(self._dump_probe(self.state, jnp.int32(offset)))
                dump_any[offset], dump_accepted[offset], dump_masks[offset] = physical, accepted, mask
        fresh = np.asarray(result["fresh"], bool)
        task_done = bool(result["task_done"])
        done = bool(result["done"])
        frame = snapshot_from_timestep(SimpleNamespace(
            state=self.state, env_cfg=self.state.env_cfg, reward=float(reward),
            done=done, info={"task_done": task_done}), action=action, actor_id=0 if action is not None else None)
        frame["agents"][0]["reach"] = (np.asarray(result["reach"]) / float(self.state.env_cfg.tile_size)).tolist()
        masks = dict(precision_required_band=self.band, fresh_dig_current=fresh[0],
                     fresh_dig_swing=fresh.any(axis=0), remaining_target=result["remaining"],
                     footprint=result["footprint"], work_cone=result["cone"],
                     pull_permission=result["permission"], dump_current=dump_masks[0],
                     dump_swing=dump_masks.any(axis=0))
        frame["maps"].update({name: np.asarray(value, bool).tolist() for name, value in masks.items()})
        admitted = bool(result["admitted"][0])
        relift = bool(result["relift"][0])
        do_kind = "unload" if loaded else ("relift" if admitted and relift else ("fresh_dig" if fresh[0].any() else "blocked"))
        dump_status = ("empty" if not loaded else "accepted_now" if dump_accepted[0]
                       else "accepted_after_swing" if dump_accepted.any()
                       else "off_zone_only" if dump_any.any() else "no_unload_at_this_base")
        completion = {name: float(value) for name, value in result["completion"].items()}
        frame["diagnostics"] = dict(
            selected_case=self.selected_case, loaded=loaded, do_kind=do_kind, message=message,
            action_had_effect=effect, accepted_unload_now=bool(dump_accepted[0]),
            accepted_unload_any=bool(dump_accepted.any()), dump_status=dump_status,
            native_unload_by_cabin_offset=dump_any.tolist(), accepted_unload_by_cabin_offset=dump_accepted.tolist(),
            fresh_cells_by_cabin_offset=fresh.sum(axis=(1, 2)).tolist(),
            current_dig_admitted=admitted, current_dig_volume=int(result["volume"][0]),
            current_dig_relift=relift, current_workspace_obstacle=bool(result["obstacle"][0]),
            base_movement_effects=np.asarray(result["base_moves"], bool).tolist(),
            step_budget=450, remaining_steps=max(0, 450-frame["step"]), exploring=self.exploring,
            metrics=dict(required=int(result["required"]), dug=int(result["dug"]),
                         disposed=completion["accepted_dump_volume"], carried=loaded,
                         illegal_spoil=completion["illegal_dump_volume"], mass=int(result["mass"]),
                         productive_workspace_cycles=int(self.state.productive_workspace_cycles),
                         **completion),
        )
        if int(result["mass"]) != self.initial_mass:
            raise RuntimeError("Native material conservation failed.")
        return frame

    def step(self, action, *, continue_after_timeout=False):
        import jax
        import jax.numpy as jnp

        if isinstance(action, bool) or not isinstance(action, int) or action not in range(8):
            raise ValueError("action must be an integer from 0 to 7.")
        if not isinstance(continue_after_timeout, bool):
            raise ValueError("continue_after_timeout must be boolean.")
        previous_frame = self.recorder.frames[-1]
        if continue_after_timeout and previous_frame["done"]:
            self.exploring = True
        if previous_frame["done"] and not self.exploring:
            raise RuntimeError("Episode ended. Reset, undo, or explicitly continue exploring beyond the episode.")
        state, reward, _, _, info = self._advance(self.state, jnp.int32(action))
        reward, info = jax.device_get((reward, info))
        if not np.isfinite(reward) or any(bool(info[name]) for name in ("target_mutation", "obstacle_mutation")) or int(info["transition_mass_residual"]):
            raise RuntimeError("Native transition integrity failed; the action was not committed.")
        effect = bool(info["action_had_effect"])
        previous = previous_frame["diagnostics"]
        if action == 7:
            message = "Waited one native action."
        elif effect:
            message = f"{ACTION_NAMES[action].capitalize()} changed the native state."
        elif action < 4 and previous["loaded"]:
            message = "Unload first: a loaded excavator cannot move its base."
        elif action < 4:
            message = "Base action blocked by terrain, clearance, or the map boundary."
        elif action == 6 and previous["loaded"]:
            message = "No native unload in this cabin direction. Try a cabin direction marked as unloadable."
        elif action == 6 and previous["current_workspace_obstacle"]:
            message = "DO blocked: the native work cone overlaps a map obstacle."
        elif action == 6 and not previous["current_dig_admitted"]:
            message = "No native fresh dig or soil pickup is admitted in this workspace."
        else:
            message = "Native action had no physical effect."
        self.states.append(state)
        try:
            frame = self._frame(reward=reward, action=action, effect=effect, message=message)
        except Exception:
            self.states.pop()
            raise
        self.actions.append(action)
        self.recorder.frames.append(frame)
        return frame

    def undo(self):
        if not self.actions:
            raise RuntimeError("Already at the saved initial state.")
        self.actions.pop()
        self.states.pop()
        self.recorder.frames.pop()
        self.exploring = bool(self.recorder.frames[-1]["diagnostics"]["exploring"])
        return self.payload()

    def continue_exploring(self):
        if not self.recorder.frames[-1]["done"]:
            raise RuntimeError("The native episode has not ended.")
        self.exploring = True
        return self.payload()

    def export(self):
        import json
        import jax

        name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        folder = self.output / f"{name}_slot{self.selected_case['case_id']}"
        folder.mkdir(parents=True, exist_ok=False)
        replay_path = self.recorder.save(folder / "replay.json.gz")
        actions_path = folder / "actions.json"
        actions_path.write_text(json.dumps(dict(metadata=self.recorder.metadata, actions=self.actions), indent=2) + "\n")
        states_path = folder / "states.pkl"
        with states_path.open("wb") as stream:
            pickle.dump(dict(metadata=self.recorder.metadata, actions=self.actions,
                             states=jax.tree_util.tree_map(lambda value: np.asarray(value), jax.device_get(self.states))),
                        stream, protocol=pickle.HIGHEST_PROTOCOL)
        return dict(replay=self.recorder.to_dict(), paths=dict(replay=str(replay_path),
                    native_states=str(states_path), actions=str(actions_path)))

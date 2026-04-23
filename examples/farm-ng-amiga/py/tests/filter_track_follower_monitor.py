from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig
from farm_ng.core.events_file_reader import proto_from_json_file
from farm_ng.filter.filter_pb2 import FilterState
from farm_ng.track.track_pb2 import TrackFollowerState


def load_config(path: Path) -> EventServiceConfig:
    return proto_from_json_file(path, EventServiceConfig())


def safe_get_filter_xy(state: FilterState) -> tuple[float, float]:
    t = state.pose.a_from_b.translation
    if hasattr(t, "x"):
        return float(t.x), float(t.y)
    return float(t[0]), float(t[1])


def summarize_filter_state(state: FilterState) -> str:
    x, y = safe_get_filter_xy(state)
    heading = float(state.heading)

    lin_vel = 0.0
    ang_vel = 0.0
    try:
        lin_vel = float(state.pose.tangent_of_b_in_a.linear_velocity.x)
    except Exception:
        pass
    try:
        ang_vel = float(state.pose.tangent_of_b_in_a.angular_velocity.z)
    except Exception:
        pass

    uncertainty = []
    try:
        uncertainty = list(state.uncertainty_diagonal.data)
    except Exception:
        pass

    divs = []
    try:
        divs = [str(d) for d in state.divergence_criteria]
    except Exception:
        pass

    lines = []
    lines.append("========== FILTER ==========")
    lines.append(f"pos: x={x:.3f}, y={y:.3f}")
    lines.append(f"heading: {heading:.3f} rad")
    lines.append(f"velocity: linear={lin_vel:.4f}, angular={ang_vel:.4f}")

    if uncertainty:
        joined = ", ".join(f"{u:.5f}" for u in uncertainty)
        lines.append(f"uncertainty_diagonal: [{joined}]")
    else:
        lines.append("uncertainty_diagonal: <unavailable>")

    if divs:
        lines.append("divergence_criteria:")
        for d in divs:
            lines.append(f"  - {d}")
    else:
        lines.append("divergence_criteria: NONE")

    ready = len(divs) == 0
    lines.append(f"filter_ready: {'YES' if ready else 'NO'}")

    return "\n".join(lines)


def enum_name(value) -> str:
    try:
        return value.name
    except Exception:
        return str(value)


def summarize_track_follower_state(state: TrackFollowerState) -> str:
    lines = []
    lines.append("====== TRACK FOLLOWER ======")

    try:
        track_status = enum_name(state.status.track_status)
    except Exception:
        track_status = "<unavailable>"

    try:
        driving_direction = enum_name(state.status.driving_direction)
    except Exception:
        driving_direction = "<unavailable>"

    try:
        waypoint_order = enum_name(state.status.waypoint_order)
    except Exception:
        waypoint_order = "<unavailable>"

    lines.append(f"track_status: {track_status}")
    lines.append(f"driving_direction: {driving_direction}")
    lines.append(f"waypoint_order: {waypoint_order}")

    failure_modes = []
    try:
        failure_modes = [enum_name(f) for f in state.status.robot_status.failure_modes]
    except Exception:
        pass

    if failure_modes:
        lines.append("failure_modes:")
        for f in failure_modes:
            lines.append(f"  - {f}")
    else:
        lines.append("failure_modes: NONE")

    try:
        p = state.progress
        lines.append(
            f"progress: goal_waypoint_index={p.goal_waypoint_index}, "
            f"track_size={p.track_size}"
        )
        lines.append(
            f"distance: remaining={p.distance_remaining:.3f} m, "
            f"total={p.distance_total:.3f} m"
        )
        lines.append(
            f"duration: remaining={p.duration_remaining:.3f} s, "
            f"total={p.duration_total:.3f} s"
        )
    except Exception:
        lines.append("progress: <unavailable>")

    try:
        wfr = state.poses.world_from_robot.a_from_b.translation
        if hasattr(wfr, "x"):
            wx, wy = float(wfr.x), float(wfr.y)
        else:
            wx, wy = float(wfr[0]), float(wfr[1])
        lines.append(f"world_from_robot: x={wx:.3f}, y={wy:.3f}")
    except Exception:
        lines.append("world_from_robot: <unavailable>")

    try:
        rfg = state.poses.robot_from_goal.a_from_b.translation
        if hasattr(rfg, "x"):
            gx, gy = float(rfg.x), float(rfg.y)
        else:
            gx, gy = float(rfg[0]), float(rfg[1])
        lines.append(f"robot_from_goal: x={gx:.3f}, y={gy:.3f}")
    except Exception:
        lines.append("robot_from_goal: <unavailable>")

    try:
        rcw = state.poses.robot_from_closest_waypoint.a_from_b.translation
        if hasattr(rcw, "x"):
            cx, cy = float(rcw.x), float(rcw.y)
        else:
            cx, cy = float(rcw[0]), float(rcw[1])
        lines.append(f"robot_from_closest_waypoint: x={cx:.3f}, y={cy:.3f}")
    except Exception:
        lines.append("robot_from_closest_waypoint: <unavailable>")

    try:
        lin_cmd = float(state.commands.linear_velocity.x)
    except Exception:
        lin_cmd = 0.0

    try:
        ang_cmd = float(state.commands.angular_velocity.z)
    except Exception:
        ang_cmd = 0.0

    lines.append(f"commands: linear={lin_cmd:.4f}, angular={ang_cmd:.4f}")

    follower_ready = len(failure_modes) == 0
    lines.append(f"track_follower_ready: {'YES' if follower_ready else 'NO'}")

    return "\n".join(lines)


async def monitor_filter(filter_config_path: Path):
    config = load_config(filter_config_path)
    client = EventClient(config)

    async for event, msg in client.subscribe(config.subscriptions[0], decode=True):
        if isinstance(msg, FilterState):
            print()
            print(summarize_filter_state(msg))
            print()


async def monitor_track_follower(track_follower_config_path: Path):
    config = load_config(track_follower_config_path)
    client = EventClient(config)

    async for event, msg in client.subscribe(config.subscriptions[0], decode=True):
        if isinstance(msg, TrackFollowerState):
            print()
            print(summarize_track_follower_state(msg))
            print()


async def combined_monitor(filter_config_path: Path, track_follower_config_path: Path):
    filter_config = load_config(filter_config_path)
    tf_config = load_config(track_follower_config_path)

    filter_client = EventClient(filter_config)
    tf_client = EventClient(tf_config)

    latest_filter: FilterState | None = None
    latest_tf: TrackFollowerState | None = None

    async def filter_task():
        nonlocal latest_filter
        async for event, msg in filter_client.subscribe(filter_config.subscriptions[0], decode=True):
            if isinstance(msg, FilterState):
                latest_filter = msg

    async def tf_task():
        nonlocal latest_tf
        async for event, msg in tf_client.subscribe(tf_config.subscriptions[0], decode=True):
            if isinstance(msg, TrackFollowerState):
                latest_tf = msg

    async def display_task():
        while True:
            print("\033[2J\033[H", end="")  # clear screen
            print("Farm-ng Follow Readiness Monitor")
            print("Press Ctrl+C to stop.\n")

            if latest_filter is not None:
                print(summarize_filter_state(latest_filter))
            else:
                print("========== FILTER ==========")
                print("No FilterState received yet.\n")

            print()

            if latest_tf is not None:
                print(summarize_track_follower_state(latest_tf))
            else:
                print("====== TRACK FOLLOWER ======")
                print("No TrackFollowerState received yet.\n")

            print()
            print("READY CHECK")
            print("-----------")

            filter_ready = False
            tf_ready = False

            if latest_filter is not None:
                try:
                    filter_ready = len(latest_filter.divergence_criteria) == 0
                except Exception:
                    filter_ready = False

            if latest_tf is not None:
                try:
                    tf_ready = len(latest_tf.status.robot_status.failure_modes) == 0
                except Exception:
                    tf_ready = False

            overall_ready = filter_ready and tf_ready
            print(f"Filter ready:         {'YES' if filter_ready else 'NO'}")
            print(f"Track follower ready: {'YES' if tf_ready else 'NO'}")
            print(f"Overall ready:        {'YES' if overall_ready else 'NO'}")

            await asyncio.sleep(0.5)

    await asyncio.gather(filter_task(), tf_task(), display_task())


def build_parser():
    parser = argparse.ArgumentParser(description="Monitor Filter and Track Follower readiness.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_filter = subparsers.add_parser("filter", help="Monitor FilterState only.")
    p_filter.add_argument("--filter-config", type=Path, required=True)

    p_tf = subparsers.add_parser("track-follower", help="Monitor TrackFollowerState only.")
    p_tf.add_argument("--track-follower-config", type=Path, required=True)

    p_both = subparsers.add_parser("both", help="Monitor both Filter and Track Follower.")
    p_both.add_argument("--filter-config", type=Path, required=True)
    p_both.add_argument("--track-follower-config", type=Path, required=True)

    return parser


async def async_main(args):
    if args.command == "filter":
        await monitor_filter(args.filter_config)
    elif args.command == "track-follower":
        await monitor_track_follower(args.track_follower_config)
    elif args.command == "both":
        await combined_monitor(args.filter_config, args.track_follower_config)


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
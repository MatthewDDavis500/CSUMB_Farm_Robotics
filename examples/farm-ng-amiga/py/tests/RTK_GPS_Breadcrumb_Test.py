from __future__ import annotations

import argparse
import asyncio
import math
from pathlib import Path

from google.protobuf.empty_pb2 import Empty
from google.protobuf.descriptor import FieldDescriptor

from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig
from farm_ng.core.events_file_reader import proto_from_json_file
from farm_ng.core.events_file_writer import proto_to_json_file
from farm_ng.filter.filter_pb2 import FilterState
from farm_ng.track.track_pb2 import Track
from farm_ng.track.track_pb2 import TrackFollowRequest
from farm_ng.canbus.canbus_pb2 import Twist2d
from farm_ng_core_pybind import Isometry3F64
from farm_ng_core_pybind import Pose3F64


POSE_SPACING_M = 0.5

ALIGNMENT_THRESHOLD_RAD = 0.20
ALIGNMENT_MAX_ANGULAR = 0.6
ALIGNMENT_KP = 1.2
ALIGNMENT_TIMEOUT_S = 60.0
ALIGNMENT_HOLD_S = 0.5
CONTROL_DT = 0.1


def load_config(path: Path) -> EventServiceConfig:
    return proto_from_json_file(path, EventServiceConfig())


def override_ip_in_string(value: str, ip: str) -> str:
    value = value.strip()

    if not value:
        return ip

    if "://" in value:
        scheme, rest = value.split("://", 1)
        if ":" in rest:
            host_part, port_part = rest.rsplit(":", 1)
            if port_part.isdigit():
                return f"{scheme}://{ip}:{port_part}"
        return f"{scheme}://{ip}"

    if ":" in value:
        host_part, port_part = value.rsplit(":", 1)
        if port_part.isdigit():
            return f"{ip}:{port_part}"

    return ip


def override_ip_in_proto(message, ip: str):
    ip_field_names = {"host", "hostname", "ip", "address"}

    for field in message.DESCRIPTOR.fields:
        if field.label == FieldDescriptor.LABEL_REPEATED:
            repeated_value = getattr(message, field.name)

            if field.type == FieldDescriptor.TYPE_MESSAGE:
                for item in repeated_value:
                    override_ip_in_proto(item, ip)

            elif field.type == FieldDescriptor.TYPE_STRING and field.name.lower() in ip_field_names:
                for i in range(len(repeated_value)):
                    repeated_value[i] = override_ip_in_string(repeated_value[i], ip)

        else:
            if field.type == FieldDescriptor.TYPE_MESSAGE:
                if message.HasField(field.name):
                    override_ip_in_proto(getattr(message, field.name), ip)

            elif field.type == FieldDescriptor.TYPE_STRING and field.name.lower() in ip_field_names:
                current_value = getattr(message, field.name)
                setattr(message, field.name, override_ip_in_string(current_value, ip))


def load_config_with_ip(path: Path, ip: str) -> EventServiceConfig:
    config = load_config(path)
    override_ip_in_proto(config, ip)
    return config


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def distance_xy(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)


def safe_translation_xy(translation) -> tuple[float, float]:
    if hasattr(translation, "x"):
        return float(translation.x), float(translation.y)
    return float(translation[0]), float(translation[1])


def get_xy_and_heading_from_filter_state(state: FilterState) -> tuple[float, float, float]:
    x, y = safe_translation_xy(state.pose.a_from_b.translation)
    heading = float(state.heading)
    return x, y, heading


def pose3_from_filter_state(state: FilterState) -> Pose3F64:
    return Pose3F64.from_proto(state.pose)


def get_xy_from_pose_proto(pose_proto) -> tuple[float, float]:
    pose = Pose3F64.from_proto(pose_proto)
    return safe_translation_xy(pose.a_from_b.translation)


def get_heading_from_pose3(pose: Pose3F64) -> float:
    q = pose.a_from_b.rotation.unit_quaternion

    if hasattr(q.imag, "__getitem__"):
        x = float(q.imag[0])
        y = float(q.imag[1])
        z = float(q.imag[2])
    else:
        x = float(getattr(q.imag, "x", 0.0))
        y = float(getattr(q.imag, "y", 0.0))
        z = float(getattr(q.imag, "z", 0.0))

    w = float(q.real)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def reverse_track_native(track: Track) -> Track:
    reversed_poses = []
    for pose_proto in reversed(track.waypoints):
        pose = Pose3F64.from_proto(pose_proto)
        reversed_pose = Pose3F64(
            a_from_b=pose.a_from_b * Isometry3F64.Rz(math.pi),
            frame_a=pose.frame_a,
            frame_b=pose.frame_b,
            tangent_of_b_in_a=pose.tangent_of_b_in_a,
        )
        reversed_poses.append(reversed_pose.to_proto())

    return Track(waypoints=reversed_poses)


def load_track(track_path: Path) -> Track:
    return proto_from_json_file(track_path, Track())


def save_track(track: Track, track_path: Path):
    ok = proto_to_json_file(track_path, track)
    if not ok:
        raise RuntimeError(f"Failed to write track to {track_path}")


def print_track_summary(track: Track, label: str):
    print("")
    print(f"========== {label} ==========")
    print(f"Waypoints: {len(track.waypoints)}")
    if len(track.waypoints) > 0:
        first_pose = Pose3F64.from_proto(track.waypoints[0])
        last_pose = Pose3F64.from_proto(track.waypoints[-1])

        fx, fy = safe_translation_xy(first_pose.a_from_b.translation)
        lx, ly = safe_translation_xy(last_pose.a_from_b.translation)

        print(f"First waypoint: x={fx:.3f}, y={fy:.3f}")
        print(f"Last waypoint:  x={lx:.3f}, y={ly:.3f}")
    print("")


def find_nearest_waypoint_index(track: Track, robot_x: float, robot_y: float) -> int:
    if len(track.waypoints) == 0:
        raise ValueError("Track has no waypoints.")

    best_idx = 0
    best_dist = float("inf")

    for i, pose_proto in enumerate(track.waypoints):
        x, y = get_xy_from_pose_proto(pose_proto)
        d = distance_xy(robot_x, robot_y, x, y)
        if d < best_dist:
            best_dist = d
            best_idx = i

    return best_idx


def compute_local_path_heading(track: Track, idx: int) -> float:
    n = len(track.waypoints)
    if n < 2:
        pose = Pose3F64.from_proto(track.waypoints[idx])
        return get_heading_from_pose3(pose)

    if idx < n - 1:
        x1, y1 = get_xy_from_pose_proto(track.waypoints[idx])
        x2, y2 = get_xy_from_pose_proto(track.waypoints[idx + 1])
        return math.atan2(y2 - y1, x2 - x1)

    x1, y1 = get_xy_from_pose_proto(track.waypoints[idx - 1])
    x2, y2 = get_xy_from_pose_proto(track.waypoints[idx])
    return math.atan2(y2 - y1, x2 - x1)


def trim_track_from_index(track: Track, start_idx: int) -> Track:
    trimmed = Track()
    for pose_proto in track.waypoints[start_idx:]:
        trimmed.waypoints.add().CopyFrom(pose_proto)
    return trimmed


class FilterStateCache:
    def __init__(self):
        self.latest_state: FilterState | None = None
        self._task: asyncio.Task | None = None

    async def start(self, filter_client: EventClient, filter_config: EventServiceConfig):
        async def _runner():
            async for event, msg in filter_client.subscribe(filter_config.subscriptions[0], decode=True):
                if isinstance(msg, FilterState):
                    self.latest_state = msg

        self._task = asyncio.create_task(_runner())

    async def wait_until_ready(self, timeout_s: float = 10.0) -> FilterState:
        start = asyncio.get_running_loop().time()
        while self.latest_state is None:
            if asyncio.get_running_loop().time() - start > timeout_s:
                raise TimeoutError("[FILTER] Timed out waiting for first FilterState.")
            await asyncio.sleep(0.05)
        return self.latest_state

    def get(self) -> FilterState:
        if self.latest_state is None:
            raise RuntimeError("[FILTER] No cached FilterState available.")
        return self.latest_state

    async def stop(self):
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


async def send_twist_command(canbus_client: EventClient, linear: float, angular: float):
    twist = Twist2d()
    twist.linear_velocity_x = float(linear)
    twist.angular_velocity = float(angular)
    await canbus_client.request_reply("/twist", twist)


async def stop_robot(canbus_client: EventClient):
    await send_twist_command(canbus_client, 0.0, 0.0)


async def set_track(track_follower_client: EventClient, track: Track):
    await track_follower_client.request_reply(
        "/set_track",
        TrackFollowRequest(track=track),
    )


async def start_track_follower(track_follower_client: EventClient):
    await track_follower_client.request_reply("/start", Empty())


async def pause_track_follower(track_follower_client: EventClient):
    await track_follower_client.request_reply("/pause", Empty())


async def resume_track_follower(track_follower_client: EventClient):
    await track_follower_client.request_reply("/resume", Empty())


async def cancel_track_follower(track_follower_client: EventClient):
    await track_follower_client.request_reply("/cancel", Empty())


async def align_to_heading(
    filter_cache: FilterStateCache,
    canbus_client: EventClient,
    target_heading: float,
    threshold_rad: float = ALIGNMENT_THRESHOLD_RAD,
    timeout_s: float = ALIGNMENT_TIMEOUT_S,
):
    print("")
    print("[ALIGN] Starting alignment...")
    print(f"[ALIGN] Target heading: {target_heading:.3f} rad")
    print("")

    start_time = asyncio.get_running_loop().time()
    within_threshold_since = None

    while True:
        now = asyncio.get_running_loop().time()
        if now - start_time > timeout_s:
            await stop_robot(canbus_client)
            raise TimeoutError("[ALIGN] Timed out before reaching heading alignment.")

        state = filter_cache.get()
        current_heading = float(state.heading)
        error = normalize_angle(target_heading - current_heading)

        print(
            f"[ALIGN] current={current_heading:.3f}, "
            f"target={target_heading:.3f}, "
            f"error={error:.3f}"
        )

        if abs(error) <= threshold_rad:
            await stop_robot(canbus_client)
            if within_threshold_since is None:
                within_threshold_since = now

            if now - within_threshold_since >= ALIGNMENT_HOLD_S:
                print("[ALIGN] Alignment complete.")
                return
        else:
            within_threshold_since = None
            angular_cmd = max(
                -ALIGNMENT_MAX_ANGULAR,
                min(ALIGNMENT_MAX_ANGULAR, ALIGNMENT_KP * error)
            )
            await send_twist_command(canbus_client, 0.0, angular_cmd)

        await asyncio.sleep(CONTROL_DT)


async def record_track_from_filter_state(
    filter_client: EventClient,
    filter_config: EventServiceConfig,
    output_track_path: Path,
    pose_spacing_m: float = POSE_SPACING_M,
):
    track = Track()

    last_x = None
    last_y = None
    waypoint_count = 0

    print("")
    print("[RECORD] Recording native Track from Filter /state")
    print("[RECORD] Drive the robot manually.")
    print("[RECORD] Press Ctrl+C when done.")
    print(f"[RECORD] Pose spacing: {pose_spacing_m:.2f} m")
    print("")

    try:
        async for event, msg in filter_client.subscribe(filter_config.subscriptions[0], decode=True):
            if not isinstance(msg, FilterState):
                continue

            x, y, heading = get_xy_and_heading_from_filter_state(msg)

            if last_x is None:
                pose = pose3_from_filter_state(msg)
                track.waypoints.add().CopyFrom(pose.to_proto())
                last_x, last_y = x, y
                waypoint_count += 1
                proto_to_json_file(output_track_path, track)
                print(
                    f"[RECORD] Waypoint {waypoint_count - 1}: "
                    f"x={x:.3f}, y={y:.3f}, heading={heading:.3f}"
                )
                continue

            dist = distance_xy(last_x, last_y, x, y)
            if dist >= pose_spacing_m:
                pose = pose3_from_filter_state(msg)
                track.waypoints.add().CopyFrom(pose.to_proto())
                last_x, last_y = x, y
                waypoint_count += 1
                proto_to_json_file(output_track_path, track)
                print(
                    f"[RECORD] Waypoint {waypoint_count - 1}: "
                    f"x={x:.3f}, y={y:.3f}, heading={heading:.3f}, step={dist:.3f} m"
                )

    except KeyboardInterrupt:
        print("")
        print("[RECORD] Stopped by user.")
        print(f"[RECORD] Saved track to: {output_track_path}")
        print(f"[RECORD] Total waypoints: {len(track.waypoints)}")
        return track

    return track


async def follow_track_file(
    track_follower_client: EventClient,
    track_path: Path,
    reverse: bool,
    filter_client: EventClient,
    filter_config: EventServiceConfig,
    canbus_client: EventClient,
):
    track = load_track(track_path)
    print_track_summary(track, "LOADED TRACK")

    if reverse:
        track = reverse_track_native(track)
        reversed_path = track_path.with_name(track_path.stem + "_reversed.json")
        save_track(track, reversed_path)
        print(f"[FOLLOW] Reversed track saved to {reversed_path}")
        print_track_summary(track, "REVERSED TRACK")

    if len(track.waypoints) == 0:
        raise ValueError("Track has no waypoints.")

    filter_cache = FilterStateCache()
    await filter_cache.start(filter_client, filter_config)
    await filter_cache.wait_until_ready(timeout_s=10.0)

    state = filter_cache.get()
    robot_x, robot_y, robot_heading = get_xy_and_heading_from_filter_state(state)

    nearest_idx = find_nearest_waypoint_index(track, robot_x, robot_y)
    target_heading = compute_local_path_heading(track, nearest_idx)

    print(f"[FOLLOW] Robot pose: x={robot_x:.3f}, y={robot_y:.3f}, heading={robot_heading:.3f}")
    print(f"[FOLLOW] Nearest waypoint index: {nearest_idx}")
    print(f"[FOLLOW] Local path heading at nearest waypoint: {target_heading:.3f}")

    await align_to_heading(
        filter_cache=filter_cache,
        canbus_client=canbus_client,
        target_heading=target_heading,
    )

    trimmed_track = trim_track_from_index(track, nearest_idx)
    print_track_summary(trimmed_track, "TRIMMED TRACK")

    print("[FOLLOW] Sending trimmed track to Track Follower...")
    await set_track(track_follower_client, trimmed_track)

    print("[FOLLOW] Starting Track Follower...")
    await start_track_follower(track_follower_client)

    print("[FOLLOW] Track Follower start command sent.")

    await filter_cache.stop()


async def learn_and_follow(
    filter_client: EventClient,
    filter_config: EventServiceConfig,
    track_follower_client: EventClient,
    canbus_client: EventClient,
    output_track_path: Path,
    reverse_for_return: bool,
    pose_spacing_m: float,
):
    recorded_track = await record_track_from_filter_state(
        filter_client=filter_client,
        filter_config=filter_config,
        output_track_path=output_track_path,
        pose_spacing_m=pose_spacing_m,
    )

    if len(recorded_track.waypoints) == 0:
        print("[ERROR] No waypoints recorded.")
        return

    print_track_summary(recorded_track, "RECORDED TRACK")

    track_to_follow = recorded_track
    if reverse_for_return:
        track_to_follow = reverse_track_native(recorded_track)
        reversed_path = output_track_path.with_name(output_track_path.stem + "_reversed.json")
        save_track(track_to_follow, reversed_path)
        print(f"[FLOW] Reversed return track saved to {reversed_path}")
        print_track_summary(track_to_follow, "RETURN TRACK")

    filter_cache = FilterStateCache()
    await filter_cache.start(filter_client, filter_config)
    await filter_cache.wait_until_ready(timeout_s=10.0)

    state = filter_cache.get()
    robot_x, robot_y, robot_heading = get_xy_and_heading_from_filter_state(state)

    nearest_idx = find_nearest_waypoint_index(track_to_follow, robot_x, robot_y)
    target_heading = compute_local_path_heading(track_to_follow, nearest_idx)

    print(f"[FLOW] Robot pose: x={robot_x:.3f}, y={robot_y:.3f}, heading={robot_heading:.3f}")
    print(f"[FLOW] Nearest waypoint index: {nearest_idx}")
    print(f"[FLOW] Local path heading at nearest waypoint: {target_heading:.3f}")

    await align_to_heading(
        filter_cache=filter_cache,
        canbus_client=canbus_client,
        target_heading=target_heading,
    )

    trimmed_track = trim_track_from_index(track_to_follow, nearest_idx)
    print_track_summary(trimmed_track, "TRIMMED TRACK")

    print("[FLOW] Sending trimmed track to Track Follower...")
    await set_track(track_follower_client, trimmed_track)

    print("[FLOW] Starting Track Follower...")
    await start_track_follower(track_follower_client)

    print("[FLOW] Track Follower start command sent.")

    await filter_cache.stop()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Native Farm-ng breadcrumb recorder using Filter /state, nearest-waypoint alignment, and Track Follower."
    )
    parser.add_argument("--ip", type=str, required=True, help="IP address to apply to all service configs at runtime.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_record = subparsers.add_parser("record", help="Record a Track from Filter /state.")
    p_record.add_argument("--filter-config", type=Path, required=True)
    p_record.add_argument("--output-track", type=Path, required=True)
    p_record.add_argument("--pose-spacing", type=float, default=POSE_SPACING_M)

    p_follow = subparsers.add_parser("follow", help="Align to nearest waypoint direction and follow.")
    p_follow.add_argument("--track-follower-config", type=Path, required=True)
    p_follow.add_argument("--filter-config", type=Path, required=True)
    p_follow.add_argument("--canbus-config", type=Path, required=True)
    p_follow.add_argument("--track", type=Path, required=True)
    p_follow.add_argument("--reverse", action="store_true")

    p_flow = subparsers.add_parser(
        "learn-and-follow",
        help="Record a Track, align to nearest waypoint direction, then follow it.",
    )
    p_flow.add_argument("--filter-config", type=Path, required=True)
    p_flow.add_argument("--track-follower-config", type=Path, required=True)
    p_flow.add_argument("--canbus-config", type=Path, required=True)
    p_flow.add_argument("--output-track", type=Path, required=True)
    p_flow.add_argument("--reverse-for-return", action="store_true")
    p_flow.add_argument("--pose-spacing", type=float, default=POSE_SPACING_M)

    p_pause = subparsers.add_parser("pause", help="Pause Track Follower.")
    p_pause.add_argument("--track-follower-config", type=Path, required=True)

    p_resume = subparsers.add_parser("resume", help="Resume Track Follower.")
    p_resume.add_argument("--track-follower-config", type=Path, required=True)

    p_cancel = subparsers.add_parser("cancel", help="Cancel Track Follower.")
    p_cancel.add_argument("--track-follower-config", type=Path, required=True)

    return parser


async def async_main(args):
    if args.command == "record":
        filter_config = load_config_with_ip(args.filter_config, args.ip)
        filter_client = EventClient(filter_config)
        await record_track_from_filter_state(
            filter_client=filter_client,
            filter_config=filter_config,
            output_track_path=args.output_track,
            pose_spacing_m=args.pose_spacing,
        )

    elif args.command == "follow":
        tf_config = load_config_with_ip(args.track_follower_config, args.ip)
        filter_config = load_config_with_ip(args.filter_config, args.ip)
        canbus_config = load_config_with_ip(args.canbus_config, args.ip)

        tf_client = EventClient(tf_config)
        filter_client = EventClient(filter_config)
        canbus_client = EventClient(canbus_config)

        await follow_track_file(
            track_follower_client=tf_client,
            track_path=args.track,
            reverse=args.reverse,
            filter_client=filter_client,
            filter_config=filter_config,
            canbus_client=canbus_client,
        )

    elif args.command == "learn-and-follow":
        filter_config = load_config_with_ip(args.filter_config, args.ip)
        tf_config = load_config_with_ip(args.track_follower_config, args.ip)
        canbus_config = load_config_with_ip(args.canbus_config, args.ip)

        filter_client = EventClient(filter_config)
        tf_client = EventClient(tf_config)
        canbus_client = EventClient(canbus_config)

        await learn_and_follow(
            filter_client=filter_client,
            filter_config=filter_config,
            track_follower_client=tf_client,
            canbus_client=canbus_client,
            output_track_path=args.output_track,
            reverse_for_return=args.reverse_for_return,
            pose_spacing_m=args.pose_spacing,
        )

    elif args.command == "pause":
        tf_config = load_config_with_ip(args.track_follower_config, args.ip)
        tf_client = EventClient(tf_config)
        await pause_track_follower(tf_client)
        print("[TF] Pause command sent.")

    elif args.command == "resume":
        tf_config = load_config_with_ip(args.track_follower_config, args.ip)
        tf_client = EventClient(tf_config)
        await resume_track_follower(tf_client)
        print("[TF] Resume command sent.")

    elif args.command == "cancel":
        tf_config = load_config_with_ip(args.track_follower_config, args.ip)
        tf_client = EventClient(tf_config)
        await cancel_track_follower(tf_client)
        print("[TF] Cancel command sent.")


def main():
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
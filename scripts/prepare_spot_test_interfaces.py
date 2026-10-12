"""Export exact Spot telemetry message definitions into a NEW test-only workspace.

No robot driver/SDK is copied. The subset uses original fully qualified message
names for serialization tests; never overlay this partial package on a robot.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

COMMIT = "818f368806938fe215b1a03389d44bf67be62587"
MESSAGES = (
    "BatteryState",
    "BatteryStateArray",
    "FootState",
    "FootStateArray",
    "Lease",
    "LeaseOwner",
    "LeaseResource",
    "LeaseArray",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spot-repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    package = args.out / "src/spot_msgs"
    (package / "msg").mkdir(parents=True)
    files = {}
    for name in ["LICENSE", *["spot_msgs/msg/" + name + ".msg" for name in MESSAGES]]:
        raw = subprocess.check_output(["git", "show", f"{COMMIT}:{name}"], cwd=args.spot_repo)
        target = package / ("LICENSE" if name == "LICENSE" else "msg/" + Path(name).name)
        target.write_bytes(raw)
        files[name] = hashlib.sha256(raw).hexdigest()
    (package / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.8)\nproject(spot_msgs)\n"
        "find_package(ament_cmake REQUIRED)\nfind_package(rosidl_default_generators REQUIRED)\n"
        "find_package(std_msgs REQUIRED)\nfind_package(geometry_msgs REQUIRED)\n"
        "find_package(builtin_interfaces REQUIRED)\nrosidl_generate_interfaces(${PROJECT_NAME}\n"
        + "".join('  "msg/' + name + '.msg"\n' for name in MESSAGES)
        + "  DEPENDENCIES std_msgs geometry_msgs builtin_interfaces)\n"
        "ament_export_dependencies(rosidl_default_runtime)\nament_package()\n"
    )
    (package / "package.xml").write_text("""<?xml version="1.0"?>
<package format="3"><name>spot_msgs</name><version>0.0.0</version>
<description>Exact source telemetry subset for isolated tests only. NOT a robot package.</description>
<maintainer email="74417491+fabiohuebel@users.noreply.github.com">Fabio Huebel</maintainer><license>MIT</license>
<buildtool_depend>ament_cmake</buildtool_depend><buildtool_depend>rosidl_default_generators</buildtool_depend>
<depend>std_msgs</depend><depend>geometry_msgs</depend><depend>builtin_interfaces</depend>
<exec_depend>rosidl_default_runtime</exec_depend><member_of_group>rosidl_interface_packages</member_of_group>
<export><build_type>ament_cmake</build_type></export></package>
""")
    receipt = dict(
        schema="spot-telemetry-test-interfaces/v1",
        commit=COMMIT,
        files=files,
        messages=list(MESSAGES),
        test_only=True,
        upstream="https://github.com/bdaiinstitute/spot_ros2",
        scope="Exact message definitions; no driver, SDK, robot connection or full interface package",
    )
    (args.out / "source-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(dict(out=str(args.out), messages=len(MESSAGES), test_only=True)))


if __name__ == "__main__":
    main()

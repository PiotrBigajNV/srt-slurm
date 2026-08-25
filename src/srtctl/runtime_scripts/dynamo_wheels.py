#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage and install exact ai-dynamo wheels for benchmark jobs."""

from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import os
import platform
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

DEFAULT_INDEX_URL = "https://pypi.org/simple"
DEFAULT_EXTRA_INDEX_URL = "https://pypi.nvidia.com"
DEFAULT_PYTHON_VERSION = "3.12"
Env = Mapping[str, str]
EXACT_REQUIREMENT_RE = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[A-Za-z0-9][A-Za-z0-9.!+_-]*)$")


def normalize_arch(arch: str) -> str:
    match arch:
        case "aarch64" | "arm64":
            return "aarch64"
        case "x86_64" | "amd64" | "x64":
            return "x86_64"
        case _:
            raise ValueError(f"unsupported Dynamo wheel arch {arch!r}; expected x86_64 or aarch64")


def _describe_file(path: Path) -> str:
    try:
        result = subprocess.run(["file", "-b", str(path)], check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return ""
    return result.stdout.strip()


def arch_from_uv_binary(source_dir: Path) -> str | None:
    """Infer compute arch from the uv binary installed by make setup ARCH=..."""
    uv_bin = source_dir / "bin" / "uv"
    if not uv_bin.exists():
        return None

    description = _describe_file(uv_bin)
    if "aarch64" in description or "ARM aarch64" in description:
        return "aarch64"
    if "x86-64" in description or "x86_64" in description:
        return "x86_64"
    return None


def _get_env(env: Env | None) -> Env:
    return os.environ if env is None else env


def detect_target_arch(source_dir: Path, env: Env | None = None) -> str:
    runtime_env = _get_env(env)
    if arch := runtime_env.get("DYNAMO_WHEEL_ARCH"):
        return normalize_arch(arch)
    if arch := runtime_env.get("SRTCTL_COMPUTE_ARCH"):
        return normalize_arch(arch)
    if arch := arch_from_uv_binary(source_dir):
        return arch
    return normalize_arch(platform.machine())


def runtime_wheel_pattern(version: str, arch: str) -> str:
    return f"ai_dynamo_runtime-{version}-*{arch}.whl"


def extra_wheel_specs(env: Env) -> list[str]:
    """Return exact, comma-separated wheel requirements requested by the config."""
    raw = env.get("DYNAMO_EXTRA_WHEEL_SPECS", "")
    specs = [item.strip() for item in raw.split(",") if item.strip()]
    for spec in specs:
        if EXACT_REQUIREMENT_RE.fullmatch(spec) is None:
            raise SystemExit(
                f"ERROR: DYNAMO_EXTRA_WHEEL_SPECS entries must be exact requirements like 'nixl==1.3.2'; got {spec!r}"
            )
    return specs


def wheel_pattern_for_spec(spec: str, arch: str) -> str:
    """Build a permissive wheel filename glob for an exact requirement."""
    normalize_arch(arch)
    match = EXACT_REQUIREMENT_RE.fullmatch(spec)
    if match is None:
        raise ValueError(f"invalid exact wheel requirement {spec!r}")
    normalized_name = re.sub(r"[-_.]+", "_", match.group("name"))
    version = match.group("version")
    return f"{normalized_name}-{version}-*.whl"


def _find_extra_wheel(wheel_dirs: list[Path], spec: str, arch: str) -> Path | None:
    """Find either a pure-Python wheel or a wheel for the requested architecture."""
    pattern = wheel_pattern_for_spec(spec, arch)
    arch_suffix = f"_{normalize_arch(arch)}.whl"
    for wheel_dir in wheel_dirs:
        if not wheel_dir.is_dir():
            continue
        for match in wheel_dir.glob(pattern):
            if match.name.endswith("-none-any.whl") or match.name.endswith(arch_suffix):
                return match
    return None


def platform_args_for_arch(arch: str) -> list[str]:
    match arch:
        case "aarch64":
            platforms = ["manylinux_2_28_aarch64", "manylinux2014_aarch64"]
        case "x86_64":
            platforms = ["manylinux_2_28_x86_64", "manylinux2014_x86_64"]
        case _:
            raise ValueError(f"unsupported Dynamo wheel arch {arch!r}")

    args: list[str] = []
    for platform_tag in platforms:
        args.extend(["--platform", platform_tag])
    return args


def build_download_command(
    *,
    version: str,
    wheel_dir: Path,
    arch: str,
    python_version: str,
    index_url: str,
    extra_index_url: str,
    extra_specs: list[str] | None = None,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--no-deps",
        "--pre",
        "--only-binary=:all:",
        "--implementation",
        "cp",
        "--python-version",
        python_version,
        *platform_args_for_arch(arch),
        "--index-url",
        index_url,
        "--extra-index-url",
        extra_index_url,
        "--dest",
        str(wheel_dir),
        f"ai-dynamo-runtime=={version}",
        f"ai-dynamo=={version}",
        *(extra_specs or []),
    ]


def _find_one(wheel_dirs: list[Path], pattern: str) -> Path | None:
    for wheel_dir in wheel_dirs:
        if not wheel_dir.is_dir():
            continue
        match = next(wheel_dir.glob(pattern), None)
        if match is not None:
            return match
    return None


def _runtime_wheel_exists(wheel_dir: Path, version: str, arch: str, pattern_override: str | None = None) -> bool:
    pattern = pattern_override or runtime_wheel_pattern(version, arch)
    return next(wheel_dir.glob(pattern), None) is not None


def prefetch(env: Env | None = None) -> None:
    runtime_env = _get_env(env)
    version = runtime_env.get("DYNAMO_VERSION", "")
    if not version:
        raise SystemExit("ERROR: DYNAMO_VERSION must be set for ai-dynamo wheel prefetch")

    source_dir = Path(runtime_env.get("SRTCTL_SOURCE_DIR", os.getcwd()))
    arch = detect_target_arch(source_dir, runtime_env)
    python_version = runtime_env.get("DYNAMO_PYTHON_VERSION", DEFAULT_PYTHON_VERSION)
    wheel_name = runtime_env.get("DYNAMO_WHEEL_NAME", f"ai_dynamo-{version}-py3-none-any.whl")
    runtime_pattern = runtime_env.get("DYNAMO_RUNTIME_WHEEL_PATTERN") or runtime_wheel_pattern(version, arch)
    extra_specs = extra_wheel_specs(runtime_env)
    extra_patterns = [wheel_pattern_for_spec(spec, arch) for spec in extra_specs]
    wheel_dir = Path(runtime_env.get("DYNAMO_WHEEL_HOST_DIR", source_dir / "wheelhouse" / "dynamo"))
    wheel_path = wheel_dir / wheel_name
    lock_path = wheel_dir / f".{version}.{arch}.lock"

    wheel_dir.mkdir(parents=True, exist_ok=True)

    def wheels_are_ready() -> bool:
        return (
            wheel_path.exists()
            and _runtime_wheel_exists(wheel_dir, version, arch, runtime_pattern)
            and all(_find_extra_wheel([wheel_dir], spec, arch) is not None for spec in extra_specs)
        )

    if wheels_are_ready():
        print(f"ai-dynamo wheels already staged for {arch}: {wheel_dir}")
        return

    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if not wheels_are_ready():
            print(f"Staging ai-dynamo wheels for {arch} / CPython {python_version}: {wheel_dir}")
            subprocess.run(
                build_download_command(
                    version=version,
                    wheel_dir=wheel_dir,
                    arch=arch,
                    python_version=python_version,
                    index_url=runtime_env.get("DYNAMO_INDEX_URL", DEFAULT_INDEX_URL),
                    extra_index_url=runtime_env.get("DYNAMO_EXTRA_INDEX_URL", DEFAULT_EXTRA_INDEX_URL),
                    extra_specs=extra_specs,
                ),
                check=True,
            )

    if not wheel_path.exists():
        raise SystemExit(f"ERROR: expected {wheel_path} after download")
    if not _runtime_wheel_exists(wheel_dir, version, arch, runtime_pattern):
        raise SystemExit(f"ERROR: expected {runtime_pattern} in {wheel_dir} after download")
    for spec, pattern in zip(extra_specs, extra_patterns, strict=True):
        if _find_extra_wheel([wheel_dir], spec, arch) is None:
            raise SystemExit(f"ERROR: expected {pattern} in {wheel_dir} after download")


def _already_installed(version: str, extra_specs: list[str] | None = None) -> bool:
    requirements = [f"ai-dynamo=={version}", f"ai-dynamo-runtime=={version}", *(extra_specs or [])]
    for requirement in requirements:
        match = EXACT_REQUIREMENT_RE.fullmatch(requirement)
        if match is None:
            return False
        package = match.group("name")
        expected_version = match.group("version")
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return False
        if installed != expected_version:
            return False

    try:
        importlib.import_module("dynamo.llm")
    except Exception:  # noqa: BLE001
        return False
    return True


def install(env: Env | None = None) -> None:
    runtime_env = _get_env(env)
    version = runtime_env.get("DYNAMO_VERSION", "")
    if not version:
        raise SystemExit("ERROR: DYNAMO_VERSION must be set for ai-dynamo wheel install")

    extra_specs = extra_wheel_specs(runtime_env)
    if _already_installed(version, extra_specs):
        print(f"ai-dynamo, ai-dynamo-runtime, and requested dependency wheels for {version} already installed")
        return

    arch = normalize_arch(
        runtime_env.get("DYNAMO_WHEEL_ARCH") or runtime_env.get("SRTCTL_COMPUTE_ARCH") or platform.machine()
    )
    wheel_name = runtime_env.get("DYNAMO_WHEEL_NAME", f"ai_dynamo-{version}-py3-none-any.whl")
    runtime_pattern = runtime_env.get("DYNAMO_RUNTIME_WHEEL_PATTERN") or runtime_wheel_pattern(version, arch)
    extra_patterns = [wheel_pattern_for_spec(spec, arch) for spec in extra_specs]
    wheel_dirs = [Path(item) for item in runtime_env.get("DYNAMO_WHEEL_DIRS", "/srtctl-wheels").split()]

    dynamo_wheel = _find_one(wheel_dirs, wheel_name)
    runtime_wheel = _find_one(wheel_dirs, runtime_pattern)
    missing_extra_patterns = [
        pattern
        for spec, pattern in zip(extra_specs, extra_patterns, strict=True)
        if _find_extra_wheel(wheel_dirs, spec, arch) is None
    ]
    if dynamo_wheel is None or runtime_wheel is None or missing_extra_patterns:
        dirs = " ".join(str(path) for path in wheel_dirs)
        raise SystemExit(
            f"ERROR: exact ai-dynamo wheels for {version} were not found in {dirs}\n"
            f"ERROR: expected {wheel_name}, {runtime_pattern}, and {', '.join(extra_patterns) or 'no extra wheels'}"
        )

    find_links_args: list[str] = []
    for wheel_dir in wheel_dirs:
        if wheel_dir.is_dir():
            find_links_args.extend(["--find-links", str(wheel_dir)])

    print(f"Installing ai-dynamo-runtime, ai-dynamo {version}, and requested dependencies from local wheels")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--pre",
            "--no-deps",
            "--no-index",
            *find_links_args,
            f"ai-dynamo-runtime=={version}",
            f"ai-dynamo=={version}",
            *extra_specs,
        ],
        check=True,
    )

    importlib.import_module("dynamo.llm")
    if any(spec.startswith("nixl==") for spec in extra_specs):
        importlib.import_module("nixl._api")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prefetch", help="Download exact ai-dynamo wheels for the compute architecture")
    subparsers.add_parser("install", help="Install exact staged ai-dynamo wheels inside a container")
    args = parser.parse_args()

    if args.command == "prefetch":
        prefetch()
    elif args.command == "install":
        install()
    else:
        parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()

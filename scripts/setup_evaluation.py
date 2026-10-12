#!/usr/bin/env python3
"""Prepare the FORGE evaluation toolchain, environments, and local engine."""
from __future__ import annotations

import argparse
import json
import os
import secrets
import shlex
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION = ROOT / "sasy"
IMPLEMENTATION_URL = "https://github.com/sasy-labs/sasy.git"
IMPLEMENTATION_COMMIT = "6dedd2bc15da491d14ec07f87c0f1ef4d339efd6"
IMPLEMENTATION_REPOSITORY_ID = 1405994907


def run(argv: list[str], *, cwd: Path = ROOT, dry_run: bool = False) -> None:
    print(f"[{cwd.relative_to(ROOT) if cwd.is_relative_to(ROOT) else cwd}] "
          + shlex.join(map(str, argv)), flush=True)
    if not dry_run:
        subprocess.run(argv, cwd=cwd, check=True)


def git(*argv: str) -> str:
    return subprocess.check_output(["git", "-C", str(IMPLEMENTATION), *argv], text=True).strip()


def write_private(path: Path, text: str) -> None:
    """Create local material once without following symlinks or overwriting it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(text)


def prepare_implementation(dry_run: bool) -> None:
    if dry_run:
        print(f"Verify public repository ID {IMPLEMENTATION_REPOSITORY_ID}; pin {IMPLEMENTATION_COMMIT}")
        run(["git", "submodule", "update", "--init", "--checkout", "--", "sasy"], dry_run=True)
        run(["git", "checkout", "--detach", IMPLEMENTATION_COMMIT], cwd=IMPLEMENTATION, dry_run=True)
        return
    if IMPLEMENTATION.is_symlink():
        raise RuntimeError("Refusing a symlink implementation checkout")
    request = urllib.request.Request("https://api.github.com/repos/sasy-labs/sasy",
                                     headers={"User-Agent": "forge-evaluation-setup"})
    with urllib.request.urlopen(request, timeout=30) as response:
        identity = json.load(response)
    if identity.get("id") != IMPLEMENTATION_REPOSITORY_ID or identity.get("private") is not False:
        raise RuntimeError("Refusing implementation repository: public identity check failed")
    expected_modules = '[submodule "sasy"]\n\tpath = sasy\n\turl = ' + IMPLEMENTATION_URL + '\n'
    if (ROOT / ".gitmodules").read_text() != expected_modules:
        raise RuntimeError("Unexpected implementation submodule configuration")
    pin = subprocess.check_output(
        ["git", "-C", str(ROOT), "ls-tree", "HEAD", "--", "sasy"], text=True
    ).strip()
    if pin != f"160000 commit {IMPLEMENTATION_COMMIT}\tsasy":
        raise RuntimeError("Implementation submodule pin does not match setup")
    if not (IMPLEMENTATION / ".git").exists():
        run(["git", "-c", "submodule.sasy.url=" + IMPLEMENTATION_URL,
             "submodule", "update", "--init", "--checkout", "--", "sasy"])
    origin = git("remote", "get-url", "origin")
    if origin not in {IMPLEMENTATION_URL, "git@github.com:sasy-labs/sasy.git"}:
        raise RuntimeError("Existing sasy origin is not the public implementation repository")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Existing sasy has tracked changes; preserve them before setup")
    # Fetch only this immutable commit, never an internal remote or moving branch.
    if subprocess.run(["git", "-C", str(IMPLEMENTATION), "cat-file", "-e",
                       IMPLEMENTATION_COMMIT + "^{commit}"], capture_output=True).returncode:
        run(["git", "fetch", "origin", IMPLEMENTATION_COMMIT], cwd=IMPLEMENTATION)
    run(["git", "checkout", "--detach", IMPLEMENTATION_COMMIT], cwd=IMPLEMENTATION)
    if git("rev-parse", "HEAD") != IMPLEMENTATION_COMMIT:
        raise RuntimeError("Implementation pin verification failed")


def prepare_compatibility_links(dry_run: bool) -> None:
    """Expose public Souffle assets to historical experiment-local helpers."""
    links = {
        ROOT / "sasy-services" / "souffle": Path("../sasy/souffle"),
        ROOT / "policies" / "common_policy.dl": Path("../sasy/souffle/common_policy.dl"),
    }
    for link, target in links.items():
        if dry_run:
            print(f"Link {link.relative_to(ROOT)} -> {target}")
            continue
        if link.is_symlink() and os.readlink(link) == str(target):
            continue
        legacy = str(target).replace("../sasy/", "../vendor/sasy/", 1)
        if link.is_symlink() and os.readlink(link) == legacy:
            link.unlink()
        if link.exists() or link.is_symlink():
            raise RuntimeError(f"Refusing to replace existing compatibility asset: {link}")
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target, target_is_directory=link.name == "souffle")


def prepare_local_config(dry_run: bool) -> None:
    # First exercise the public implementation's supported local setup.
    run(["make", "init-config", "certs"], cwd=IMPLEMENTATION, dry_run=dry_run)
    if dry_run:
        print("Generate ignored local benchmark API-key/mTLS auth, role config, .env, and certificates")
        return
    config_path = ROOT / "config" / "auth" / "all.json"
    env_path = ROOT / ".env"
    if config_path.is_symlink() or env_path.is_symlink():
        raise RuntimeError("Refusing symlink local authentication settings")
    # These are fresh localhost-only credentials, never release artifacts.
    if not config_path.exists():
        if env_path.exists():
            raise RuntimeError("Existing .env with no benchmark auth config: move it aside before setup")
        suffix = secrets.token_hex(32)
        api_keys = {"developer-key": "developer", "admin-test-key": "admin"}
        config = {"type": "chain", "providers": [
            {"type": "api_key", "metadata_key": "x-api-key", "static_keys": api_keys},
            {"type": "mtls", "entity_source": "cn"},
        ]}
        write_private(config_path, json.dumps(config, indent=2) + "\n")
        # Keep the full AgentDojo key out of the environment shared by mTLS cells.
        settings = {
            "SASY_URL": "127.0.0.1:11089", "SASY_API_KEY_SUFFIX": suffix,
            "TLS_CA_PATH": str(ROOT / "certs" / "ca.crt"),
            "TLS_CERT_PATH": str(ROOT / "certs" / "admin.crt"),
            "TLS_KEY_PATH": str(ROOT / "certs" / "admin.key"),
        }
        write_private(env_path, "".join(f"{key}={value}\n" for key, value in settings.items()))
    elif not env_path.exists():
        raise RuntimeError("Existing local auth config without .env; restore matching settings")
    roles = ["reference-monitor-user", "observability-writer", "observability-reader", "openai-access"]
    entities = {
        "admin": {"roles": roles + ["admin", "credential-reader", "credential-writer", "observability-admin"]},
        "developer": {"tenant": "acme", "roles": roles},
        "jsmith": {"tenant": "acme", "roles": roles},
        "achen": {"tenant": "acme", "roles": roles},
        "medical-client": {"tenant": "malade", "roles": roles + ["fda-access", "research-access"]},
    }
    # JSON is valid YAML and avoids a third-party bootstrap dependency.
    role_path = ROOT / "config" / "auth_config.yaml"
    if not role_path.exists():
        write_private(role_path, json.dumps({"trust_domain": "forge.local", "default_tenant": "default",
                                             "entities": entities}, indent=2) + "\n")
    transform_path = ROOT / "config" / "transforms.json"
    if not transform_path.exists():
        transforms = {"transforms": {
            "inject_openai_key": {"type": "add_header", "key": "Authorization",
                                  "format": "Bearer {api_key}", "service": "openai"},
            "inject_fda_key": {"type": "add_url_param", "key": "api_key",
                               "format": "{api_key}", "service": "fda"},
            "inject_brave_key": {"type": "add_header", "key": "X-Subscription-Token",
                                 "format": "{api_key}", "service": "brave"},
        }}
        write_private(transform_path, json.dumps(transforms, indent=2) + "\n")
    credential_path = ROOT / ".env-credentials"
    if not credential_path.exists():
        write_private(credential_path, "# Provider credentials may be set here or in the process environment.\n")
    prepare_certificates()


def prepare_certificates() -> None:
    certs = ROOT / "certs"
    certs.mkdir(mode=0o700, exist_ok=True)
    names = ["reference-monitor", "admin", "medical-client", "jsmith", "achen"]
    expected = [certs / "ca.crt", certs / "ca.key"] + [certs / f"{name}.{ext}"
                for name in names for ext in ("crt", "key")]
    if all(path.is_file() and not path.is_symlink() for path in expected):
        print("Keeping existing local benchmark certificates")
        return
    if any(path.exists() or path.is_symlink() for path in expected):
        raise RuntimeError("Incomplete certificate set; restore or move certs/ aside before setup")
    # OpenSSL output contains only generated public subjects, never existing secrets.
    run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", str(certs / "ca.key"),
         "-out", str(certs / "ca.crt"), "-days", "365", "-subj", "/CN=FORGE evaluation local CA"])
    extension = certs / "localhost.ext"
    write_private(extension, "subjectAltName=DNS:localhost,IP:127.0.0.1\nextendedKeyUsage=serverAuth,clientAuth\n")
    try:
        for index, name in enumerate(names, start=1):
            csr = certs / f"{name}.csr"
            run(["openssl", "req", "-newkey", "rsa:2048", "-nodes", "-keyout", str(certs / f"{name}.key"),
                 "-out", str(csr), "-subj", f"/CN={name}"])
            run(["openssl", "x509", "-req", "-in", str(csr), "-CA", str(certs / "ca.crt"),
                 "-CAkey", str(certs / "ca.key"), "-set_serial", str(index), "-out", str(certs / f"{name}.crt"),
                 "-days", "365", "-extfile", str(extension)])
            csr.unlink()
    finally:
        extension.unlink(missing_ok=True)
        for path in certs.glob("*.key"):
            path.chmod(0o600)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--nix", action="store_true", help="Run setup inside the pinned implementation Nix development shell")
    parser.add_argument("--dry-run", action="store_true", help="Print setup actions without network or writes")
    parser.add_argument("--skip-build", action="store_true", help="Skip engine compilation")
    parser.add_argument("--skip-python", action="store_true", help="Skip evaluation environment installation")
    parser.add_argument("--skip-agentdojo", action="store_true", help="Skip the separate Python 3.12 environment")
    parser.add_argument("--build-loadgen", action="store_true", help="Also build the Rust authorization replay client")
    args = parser.parse_args()
    if args.nix:
        if not args.dry_run:
            missing = [name for name in ("git", "nix") if shutil.which(name) is None]
            if missing:
                parser.error("Missing tools: " + ", ".join(missing))
        prepare_implementation(args.dry_run)
        forwarded = [arg for arg in sys.argv[1:] if arg != "--nix"]
        run(["nix", "--extra-experimental-features", "nix-command flakes", "develop",
             str(IMPLEMENTATION), "--command", sys.executable, str(Path(__file__).resolve()),
             *forwarded], dry_run=args.dry_run)
        return 0
    required = ["git", "make", "openssl"] + ([] if args.skip_build else ["cargo"])
    if not args.skip_python:
        required.append("uv")
    if not args.dry_run:
        missing = [name for name in required if shutil.which(name) is None]
        if missing:
            parser.error("Missing tools: " + ", ".join(missing))
    prepare_implementation(args.dry_run)
    prepare_compatibility_links(args.dry_run)
    prepare_local_config(args.dry_run)
    if not args.skip_build:
        run(["make", "build-rust-dev"], cwd=IMPLEMENTATION, dry_run=args.dry_run)
    if args.build_loadgen:
        run(["cargo", "build", "--locked", "--manifest-path", "latency/loadgen/Cargo.toml"], dry_run=args.dry_run)
    if not args.skip_python:
        run(["uv", "sync", "--python", "3.11", "--all-packages"], dry_run=args.dry_run)
        if not args.skip_agentdojo:
            run(["bash", "experiments/agentdojo/setup.sh"], dry_run=args.dry_run)
    print("Setup plan complete." if args.dry_run else "Setup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

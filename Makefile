.DEFAULT_GOAL := help
PYTHON ?= python3
ARGS ?=
.PHONY: help setup serve benchmark benchmark-dry summary build-rust-dev build-loadgen
help:
	@echo "make setup          Initialize pinned SASY submodule; build engine and environments"
	@echo "make serve          Start the local engine for standalone translation experiments"
	@echo "make benchmark-dry  Inspect the experiment plan (ARGS selects cases and trials)"
	@echo "make benchmark      Run experiments (provider credentials required)"
	@echo "make summary        Summarize saved results (ARGS supplies summarizer options)"
	@echo "make build-rust-dev Rebuild sasy/target/debug/sasy"
setup:
	$(PYTHON) scripts/setup_evaluation.py $(ARGS)
build-rust-dev:
	$(MAKE) -C sasy build-rust-dev
build-loadgen:
	cargo build --locked --manifest-path latency/loadgen/Cargo.toml
serve:
	LLM_ENABLED=true uv run --env-file .env --env-file .env-credentials \
		sasy/target/debug/sasy serve --addr 127.0.0.1:11089 \
		--evaluator souffle --auth-provider config/auth/all.json \
		--auth-config config/auth_config.yaml --transforms config/transforms.json \
		--tls-cert certs/reference-monitor.crt --tls-key certs/reference-monitor.key \
		--tls-ca certs/ca.crt --data-dir outputs/local-engine/graph \
		--credential-backend memory --credentials-env-file .env-credentials \
		--allow-user-functors=unsandboxed
benchmark:
	uv run scripts/evaluation/run_benchmark.py --server-bin sasy/target/debug/sasy $(ARGS)
benchmark-dry:
	uv run scripts/evaluation/run_benchmark.py --server-bin sasy/target/debug/sasy --dry-run $(ARGS)
summary:
	uv run scripts/evaluation/summarize_benchmark.py $(ARGS)

# Translation policy validation.
export FILE POLICY SCENARIOS TRACES
.PHONY: souffle-validate policy-test policy-finalize
souffle-validate:
	@test -n "$$FILE" || { echo "Set FILE=policy.dl" >&2; exit 1; }
	uv run scripts/validate_policy.py --no-rpc "$$FILE"
policy-test:
	@test -n "$$POLICY" -a -n "$$SCENARIOS" || { echo "Set POLICY and SCENARIOS" >&2; exit 1; }
	uv run scripts/policy_test_harness.py --policy "$$POLICY" --scenarios "$$SCENARIOS"
policy-finalize:
	@test -n "$$POLICY" -a -n "$$SCENARIOS" || { echo "Set POLICY and SCENARIOS" >&2; exit 1; }
	@if test -n "$$TRACES"; then \
		uv run scripts/policy_finalize.py --policy "$$POLICY" --scenarios "$$SCENARIOS" --traces "$$TRACES"; \
	else \
		uv run scripts/policy_finalize.py --policy "$$POLICY" --scenarios "$$SCENARIOS"; \
	fi

# Vulture whitelist for torchquad.
#
# Symbols listed here are reported as unused by vulture but are actually
# reachable through mechanisms vulture cannot follow (autoray string-based
# registration, public API entry points called only by tests/downstream users).
# Each entry corresponds to a real symbol used implicitly or cross-module.
#
# Policy (CLAUDE.md rule 13): prefer deleting genuinely dead code over adding it
# here. A stale whitelist entry is itself a finding.
#
# This file is a vulture artifact, not runnable code (it is excluded from ruff);
# the ``_.attr`` references simply tell vulture the names are used.

_.get_jit_compiled_integrate  # public JIT API, exercised in tests (grid_integrator, monte_carlo)

_torch_repeat  # registered with autoray via @register_function (integration/utils.py)
_torch_expand_dims  # registered with autoray via @register_function (integration/utils.py)
_get_exp_func  # imported by tests/test_deployment.py
_get_sin_func  # imported by tests/test_deployment.py

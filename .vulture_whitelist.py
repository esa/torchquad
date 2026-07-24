# Vulture whitelist for torchquad.
#
# Symbols listed here are reported as unused by vulture but are actually
# reachable through mechanisms vulture cannot follow (autoray string-based
# registration, public API entry points called only by downstream users,
# framework callbacks). Each entry MUST correspond to a real symbol that is
# used implicitly or cross-module.
#
# Policy (CLAUDE.md rule 13 / roadmap C4): prefer deleting genuinely dead code
# over adding it here. A stale whitelist entry is itself a finding.
#
# Reference the names as attributes on a dummy object so vulture treats them
# as "used". Populate as needed when the advisory (60%) tier surfaces a real
# implicit use.

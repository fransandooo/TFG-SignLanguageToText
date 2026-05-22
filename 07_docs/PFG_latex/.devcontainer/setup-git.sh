#!/usr/bin/env bash
set -euo pipefail

# Host ~/.ssh is bind-mounted to /root/.ssh by devcontainer.json.
# Git commit signing on the host uses /home/fransando/.ssh/... — recreate that path here.
mkdir -p /home/fransando
ln -sfn /root/.ssh /home/fransando/.ssh

if [[ -f /root/.ssh/id_github_personal.pub ]]; then
    git config --global user.signingkey /root/.ssh/id_github_personal.pub
fi
if [[ -f /root/.ssh/allowed_signers ]]; then
    git config --global gpg.ssh.allowedsignersfile /root/.ssh/allowed_signers
fi

# Forward agent when the host exposes it (Linux/macOS with ssh-agent).
if [[ -n "${SSH_AUTH_SOCK:-}" && -S "${SSH_AUTH_SOCK}" ]]; then
    git config --global core.sshCommand "ssh -o ForwardAgent=yes"
fi

echo "Git SSH/signing paths configured for dev container."

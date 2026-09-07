#!/usr/bin/env bash

echo "worker-comfyui: Build version ${BUILD_VERSION:-unknown}"

# SYMLINK MODEL DIRS TO NETWORK VOLUME IF PRESENT
if [ -d /runpod-volume ]; then
    echo "worker-comfyui: Network volume detected, symlinking model dirs to /runpod-volume/models"
    for dir in diffusion_models clip vae loras controlnet; do
        mkdir -p "/runpod-volume/models/$dir"
        ln -sfn "/runpod-volume/models/$dir" "/comfyui/models/$dir"
    done
else
    echo "worker-comfyui: No network volume detected, using local model storage"
fi

# Download missing models via hf download (hf_xet chunk-based parallel transfers).
# Runs in foreground so models are guaranteed ready before ComfyUI starts.
echo "worker-comfyui: Validating models..."
/usr/local/bin/check-models.sh

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

# Ensure ComfyUI-Manager runs in offline network mode inside the container
comfy-manager-set-mode offline || echo "worker-comfyui - Could not set ComfyUI-Manager network_mode" >&2

echo "worker-comfyui: Starting ComfyUI"

# Allow operators to tweak verbosity; default is DEBUG.
: "${COMFY_LOG_LEVEL:=DEBUG}"

# Use the compatibility overlay when present so simplified mode can accept
# images[] with 1-3 references while preserving the legacy handler API.
HANDLER_PATH="/handler.py"
if [ -f /handler_multi.py ]; then
    HANDLER_PATH="/handler_multi.py"
fi

# Serve the API and don't shutdown the container
if [ "$SERVE_API_LOCALLY" == "true" ]; then
    python -u /comfyui/main.py --disable-auto-launch --disable-metadata --listen --verbose "${COMFY_LOG_LEVEL}" --log-stdout &
    echo $! > /tmp/comfyui.pid

    echo "worker-comfyui: Starting RunPod Handler (${HANDLER_PATH})"
    python -u "${HANDLER_PATH}" --rp_serve_api --rp_api_host=0.0.0.0
else
    python -u /comfyui/main.py --disable-auto-launch --disable-metadata --verbose "${COMFY_LOG_LEVEL}" --log-stdout &
    echo $! > /tmp/comfyui.pid

    echo "worker-comfyui: Starting RunPod Handler (${HANDLER_PATH})"
    python -u "${HANDLER_PATH}"
fi

#!/bin/bash
## Script to install packages on a single node in the cluster
## See setup_all.sh for more details

# Exit immediately if a command fails
set -e

# Ensure passwordless sudo is available to avoid hanging
sudo -n true 2>/dev/null || { echo "  sudo requires a password; aborting." >&2; exit 1; }

# Clean up tmp files (safe even if empty)
TMP_SIZE=$(sudo du -h -d 0 /tmp | awk '{print $1}')
sudo rm -rf /tmp/*  2>/dev/null || true
echo "  Cleaned up $TMP_SIZE of tmp files."

# create temp dirs and allow anyone to read and write
sudo mkdir -p /opt/dlami/nvme/hf_cache /opt/dlami/nvme/hf_cache/apptainer_cache /opt/dlami/nvme/hf_cache/apptainer_tmp
sudo chmod -R 777 /opt/dlami/nvme/hf_cache


# Check if apptainer is already installed
if command -v apptainer &> /dev/null
then
    echo "  Apptainer is already installed."
else
    echo "  Apptainer not found, starting installation..."
        # --- Install Apptainer ---
    # Run apt update and install software-properties-common needed for add-apt-repository
    sudo -n true 2>/dev/null || { echo "  sudo requires a password; aborting." >&2; exit 1; }
    sudo apt-get update
    sudo apt-get install -y software-properties-common

    # Add the Apptainer PPA repository
    sudo add-apt-repository -y ppa:apptainer/ppa

    # Update package list again and install apptainer
    sudo apt-get update
    sudo apt-get install -y apptainer

    echo "  Apptainer installation complete."
fi

# --- Install Apache Arrow libraries for PyArrow ---
echo "  Setting up Apache Arrow libraries..."

# Check if Arrow libraries are already installed
if dpkg -l | grep -q libarrow-dev; then
    echo "  Apache Arrow libraries already installed."
else
    echo "  Installing Apache Arrow libraries..."
    
    # Add Apache Arrow repository
    UBUNTU_CODENAME=$(lsb_release --codename --short)
    UBUNTU_ID=$(lsb_release --id --short | tr 'A-Z' 'a-z')
    
    # Download and install the Apache Arrow APT repository package
    wget -q -O /tmp/apache-arrow-apt-source.deb \
        "https://apache.jfrog.io/artifactory/arrow/${UBUNTU_ID}/apache-arrow-apt-source-latest-${UBUNTU_CODENAME}.deb"
    
    sudo dpkg -i /tmp/apache-arrow-apt-source.deb
    rm /tmp/apache-arrow-apt-source.deb
    
    # Update package list
    sudo apt-get update
    
    # Install Arrow C++ libraries and optional dependencies needed for PyArrow
    sudo apt-get install -y \
        libarrow-dev \
        libarrow-dataset-dev \
        libarrow-acero-dev \
        libparquet-dev \
        libarrow-flight-dev \
        libre2-dev \
        libthrift-dev \
        libc-ares-dev
    
    # Try to install compute-dev if it exists (may have version suffix)
    sudo apt-get install -y libarrow-compute-dev 2>/dev/null || \
        echo "  libarrow-compute-dev not found as separate package (may be included in libarrow-dev)"
    
    echo "  Apache Arrow libraries installation complete."
fi

# --- Install Rust (needed for Python 3.14t dependencies) ---
if command -v rustc &> /dev/null; then
    echo "  Rust is already installed ($(rustc --version))."
else
    echo "  Installing Rust system-wide..."
    
    # Install Rust system-wide to /usr/local
    export CARGO_HOME=/usr/local/cargo
    export RUSTUP_HOME=/usr/local/rustup
    
    sudo mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"
    sudo curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
        sudo sh -s -- -y --default-toolchain stable --no-modify-path
    
    # Add to system PATH for all users
    echo 'export PATH="/usr/local/cargo/bin:$PATH"' | sudo tee /etc/profile.d/rust.sh
    sudo chmod +x /etc/profile.d/rust.sh
    
    echo "  Rust installation complete (system-wide in /usr/local)."
    echo "  Note: Users may need to log out and back in, or run: source /etc/profile.d/rust.sh"
fi

# --- Install CMake 3.28+ (needed for PyArrow) ---
CMAKE_VERSION=$(cmake --version 2>/dev/null | head -1 | awk '{print $3}' || echo "0.0.0")
CMAKE_REQUIRED="3.25"

if [ "$(printf '%s\n' "$CMAKE_REQUIRED" "$CMAKE_VERSION" | sort -V | head -n1)" = "$CMAKE_REQUIRED" ]; then
    echo "  CMake $CMAKE_VERSION is already installed (>= $CMAKE_REQUIRED)."
else
    echo "  Installing CMake 3.28..."
    
    cd /tmp
    wget -q https://github.com/Kitware/CMake/releases/download/v3.28.1/cmake-3.28.1-linux-x86_64.sh
    bash cmake-3.28.1-linux-x86_64.sh --skip-license --prefix=/usr/local
    rm cmake-3.28.1-linux-x86_64.sh
    
    echo "  CMake installation complete ($(cmake --version | head -1))."
fi

echo "  Node setup complete!"
echo "  - Apptainer: $(apptainer --version 2>/dev/null || echo 'installed')"
echo "  - Arrow: $(dpkg -l | grep libarrow-dev | awk '{print $3}' || echo 'installed')"
echo "  - Rust: $(rustc --version 2>/dev/null || echo 'installed')"
echo "  - CMake: $(cmake --version | head -1)"
echo ""
echo "  NOTE: When installing PyArrow with uv, use these environment variables:"
echo "    export PYARROW_WITH_CUDA=0"
echo "    export PYARROW_WITH_FLIGHT=0"
echo "    export PYARROW_WITH_GANDIVA=0"
echo "    export PYARROW_WITH_AZURE=0"
echo "    export PYARROW_PARALLEL=4"
echo "  Then run: uv sync"


# Code Analysis Tools for Terminal Agent

Terminal Agent provides powerful code analysis capabilities by integrating various Language Server Protocol (LSP) tools and code indexing tools, enabling intelligent code analysis, error detection, and code navigation. This document details the installation and configuration steps for these tools.

## Table of Contents

- [Language Server Protocol (LSP) Tools](#language-server-protocol-lsp-tools)
  - [Go - gopls and golangci-lint](#go---gopls-and-golangci-lint)
  - [Rust - rust-analyzer](#rust---rust-analyzer)
  - [C++ - clangd](#c---clangd)
- [Code Search Tools](#code-search-tools)
  - [Zoekt Index and Search Server](#zoekt-index-and-search-server)
- [Configuring Terminal Agent to Use These Tools](#configuring-terminal-agent-to-use-these-tools)

## Language Server Protocol (LSP) Tools

### Go - gopls and golangci-lint

#### gopls

[gopls](https://github.com/golang/tools/tree/master/gopls) is the official language server for Go, providing code completion, error checking, formatting, and more.

##### Installation Steps

1. Ensure Go is installed (version 1.18 or higher recommended):

```bash
# Check Go version
go version

# If not installed, you can install via package manager
# macOS
brew install go

# Ubuntu/Debian
sudo apt-get install golang

# CentOS/RHEL
sudo yum install golang
```

2. Install gopls:

```bash
go install golang.org/x/tools/gopls@latest
```

3. Verify installation:

```bash
gopls version
```

#### golangci-lint

[golangci-lint](https://github.com/golangci/golangci-lint) is a fast Go linter that integrates multiple linters to detect code issues, performance concerns, and style inconsistencies.

##### Installation Steps

**Using the official installation script**:

```bash
# Install the latest version
curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b $(go env GOPATH)/bin

# Install a specific version (e.g., v1.55.2)
# curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b $(go env GOPATH)/bin v1.55.2
```

**Using package managers**:

```bash
# macOS
brew install golangci-lint

# Arch Linux
pacman -S golangci-lint

# Debian/Ubuntu
# Add PPA and install
curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b /usr/local/bin
```

**Using Go**:

```bash
go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
```

**Verify installation**:

```bash
golangci-lint --version
```

**Basic usage**:

```bash
# Run all default linters in the current directory
golangci-lint run

# Specify particular linters
golangci-lint run --enable=gosimple,govet,gofmt

# Analyze specific files or directories
golangci-lint run ./pkg/... ./cmd/...
```

### Rust - rust-analyzer

[rust-analyzer](https://rust-analyzer.github.io/) is a language server for Rust that provides intelligent code completion and analysis.

#### Installation Steps

1. Install Rust and Cargo:

```bash
# Install rustup (Rust toolchain installer)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Add Cargo to PATH
source $HOME/.cargo/env
```

2. Install rust-analyzer:

```bash
# macOS
brew install rust-analyzer

# Using rustup
rustup component add rust-analyzer

# Using cargo
cargo install --git https://github.com/rust-analyzer/rust-analyzer.git rust-analyzer
```

3. Verify installation:

```bash
rust-analyzer --version
```

### C++ - clangd

[clangd](https://clangd.llvm.org/) is an LLVM-based language server for C/C++.

#### Installation Steps

1. Install clangd:

```bash
# macOS
brew install llvm
# Add to PATH
echo 'export PATH="/usr/local/opt/llvm/bin:$PATH"' >> ~/.zshrc  # or ~/.bashrc
source ~/.zshrc  # or source ~/.bashrc

# Ubuntu/Debian
sudo apt-get install clangd-12
sudo update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-12 100

# CentOS/RHEL
sudo yum install llvm-toolset-7
scl enable llvm-toolset-7 bash
```

2. Verify installation:

```bash
clangd --version
```

## Code Search Tools

### Zoekt Index and Search Server

[Zoekt](https://github.com/sourcegraph/zoekt) is a fast code search engine that supports regular expressions and other advanced search features.

#### Installation Steps

1. Ensure Go is installed:

```bash
go version
```

2. Set up GOPATH and install zoekt-index and zoekt-webserver:

```bash
# Set GOPATH (if not already set)
export GOPATH=$HOME/go
export PATH=$PATH:$GOPATH/bin

# Add these settings to your shell configuration file for permanent effect
echo 'export GOPATH=$HOME/go' >> ~/.bashrc  # or ~/.zshrc
echo 'export PATH=$PATH:$GOPATH/bin' >> ~/.bashrc  # or ~/.zshrc
source ~/.bashrc  # or source ~/.zshrc

# Install zoekt tools
go install github.com/sourcegraph/zoekt/cmd/zoekt-index@latest
go install github.com/sourcegraph/zoekt/cmd/zoekt-webserver@latest

# Verify installation
zoekt-index -version
zoekt-webserver -version
```

## Configuring Terminal Agent to Use These Tools

After installing the above tools, Terminal Agent will automatically detect and use them for code analysis. Make sure these tools are available in your system PATH.

### Verification

1. Start Terminal Agent:

```bash
terminal-agent
```

2. Test the code analysis features:
   - For Go code analysis, open a Go file and check for linting feedback
   - For code search, use the search commands to find code across repositories
   - For LSP features, check for code completion and error highlighting

### Troubleshooting

If you encounter issues with code analysis tools:
   - Verify that the tools are correctly installed and in your PATH
   - Check Terminal Agent logs for any error messages
   - Ensure that you have created indexes for your code repositories
   - Consider increasing the index update frequency

# 代码分析功能指南

Terminal Agent 提供强大的代码分析功能，通过集成多种语言服务器协议(LSP)工具和代码索引工具，实现智能代码分析、错误检测和代码导航。本文档详细介绍这些工具的安装和配置步骤。

## 目录

- [语言服务器协议(LSP)工具](#语言服务器协议lsp工具)
  - [Go - gopls](#go---gopls)
  - [Python - flake8](#python---flake8)
  - [Rust - rust-analyzer](#rust---rust-analyzer)
  - [JavaScript/TypeScript - typescript-language-server](#javascripttypescript---typescript-language-server)
  - [C/C++ - clangd](#cc---clangd)
- [代码搜索工具](#代码搜索工具)
  - [Zoekt 索引和搜索服务器](#zoekt-索引和搜索服务器)
- [配置 Terminal Agent 使用这些工具](#配置-terminal-agent-使用这些工具)

## 语言服务器协议(LSP)工具

### Go - gopls 和 golangci-lint

#### gopls

[gopls](https://github.com/golang/tools/tree/master/gopls) 是 Go 语言的官方语言服务器，提供代码补全、错误检查、格式化等功能。

##### 安装步骤

1. 确保已安装 Go (推荐 1.18 或更高版本)：

```bash
# 检查 Go 版本
go version

# 如果未安装，可以通过包管理器安装
# macOS
brew install go

# Ubuntu/Debian
sudo apt-get install golang

# CentOS/RHEL
sudo yum install golang
```

2. 安装 gopls：

```bash
go install golang.org/x/tools/gopls@latest
```

3. 验证安装：

```bash
gopls version
```

#### golangci-lint

[golangci-lint](https://github.com/golangci/golangci-lint) 是一个快速的 Go 代码检查工具，集成了多种 linter，可以检测代码问题、性能隐患和风格不一致。

##### 安装步骤

**使用官方安装脚本**：

```bash
# 安装最新版本
curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b $(go env GOPATH)/bin

# 安装特定版本 (例如 v1.55.2)
# curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b $(go env GOPATH)/bin v1.55.2
```

**使用包管理器**：

```bash
# macOS
brew install golangci-lint

# Arch Linux
pacman -S golangci-lint

# Debian/Ubuntu
# 添加 PPA 并安装
curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b /usr/local/bin
```

**使用 Go**：

```bash
go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
```

**验证安装**：

```bash
golangci-lint --version
```

**基本使用**：

```bash
# 在当前目录运行所有默认 linter
golangci-lint run

# 指定特定 linter
golangci-lint run --enable=gosimple,govet,gofmt

# 分析特定文件或目录
golangci-lint run ./pkg/... ./cmd/...
```


### Rust - rust-analyzer

[rust-analyzer](https://rust-analyzer.github.io/) 是 Rust 语言的 LSP 实现，提供智能代码补全和分析功能。

#### 安装步骤

1. 确保已安装 Rust 和 Cargo：

```bash
# 安装 Rust 和 Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

2. 安装 rust-analyzer：

**macOS**:
```bash
brew install rust-analyzer
```

**Linux**:
```bash
# 下载预编译二进制文件
mkdir -p ~/.local/bin
curl -L https://github.com/rust-analyzer/rust-analyzer/releases/latest/download/rust-analyzer-x86_64-unknown-linux-gnu.gz | gunzip -c - > ~/.local/bin/rust-analyzer
chmod +x ~/.local/bin/rust-analyzer
```

**使用 Cargo 安装**:
```bash
cargo install --git https://github.com/rust-analyzer/rust-analyzer.git rust-analyzer
```

3. 验证安装：

```bash
rust-analyzer --version
```



### C++ - clangd

[clangd](https://clangd.llvm.org/) 是基于 LLVM 的 C/C++ 语言服务器。

#### 安装步骤

**macOS**:
```bash
brew install llvm
# 添加到 PATH
echo 'export PATH="$(brew --prefix llvm)/bin:$PATH"' >> ~/.zshrc
# 或者 ~/.bashrc
source ~/.zshrc
```

**Ubuntu/Debian**:
```bash
sudo apt-get install clangd-13
sudo update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-13 100
```

**CentOS/RHEL**:
```bash
sudo yum install clang-tools-extra
```

验证安装：
```bash
clangd --version
```

## 代码搜索工具

### Zoekt 索引和搜索服务器

[Zoekt](https://github.com/sourcegraph/zoekt) 是一个快速的代码搜索引擎，支持正则表达式和其他高级搜索功能。

#### 安装步骤

1. 确保已安装 Go：

```bash
go version
```

2. 设置 GOPATH 并安装 zoekt-index 和 zoekt-webserver：

```bash
# 设置 GOPATH (如果尚未设置)
export GOPATH=$HOME/go
export PATH=$PATH:$GOPATH/bin

# 将上述设置添加到 shell 配置文件中以便永久生效
echo 'export GOPATH=$HOME/go' >> ~/.bashrc  # 或 ~/.zshrc
echo 'export PATH=$PATH:$GOPATH/bin' >> ~/.bashrc  # 或 ~/.zshrc
source ~/.bashrc  # 或 source ~/.zshrc

# 安装 zoekt 工具
go install github.com/sourcegraph/zoekt/cmd/zoekt-index@latest
go install github.com/sourcegraph/zoekt/cmd/zoekt-webserver@latest

# 验证安装
zoekt-index -version
zoekt-webserver -version
```


## 配置 Terminal Agent 使用这些工具

安装完上述工具后，Terminal Agent 会自动检测并使用这些工具进行代码分析。确保这些工具在系统 PATH 中可用。

### 验证配置

1. 启动 Terminal Agent：

```bash
terminal-agent
```

2. 测试代码分析功能：

```
[Terminal Agent] > 分析当前目录中的 Python 代码并检查错误
```

### 常见问题排查

1. **工具未被检测到**：
   - 确认工具已正确安装并在 PATH 中
   - 尝试重启 Terminal Agent

2. **分析结果不准确**：
   - 确保已为正确的语言安装了相应的 LSP 工具
   - 检查工具版本是否过旧

3. **Zoekt 搜索速度慢**：
   - 确保已为代码库创建了索引
   - 考虑增加索引更新频率


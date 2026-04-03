# Git Operations Reference

## What we did in this session (in order)

### 1. Install GitHub CLI
```powershell
winget install --id GitHub.cli -e --accept-source-agreements --accept-package-agreements
```
GitHub CLI (`gh`) lets you create and manage GitHub repos from the terminal.
Since winget does not update PATH in the current shell, use the full path until you restart:
```powershell
& "C:\Program Files\GitHub CLI\gh.exe" auth login
```

### 2. Authenticate with GitHub
```powershell
& "C:\Program Files\GitHub CLI\gh.exe" auth login
```
- Choose: `GitHub.com` → `HTTPS` → `Login with a web browser`
- Completes OAuth in browser and stores token in keyring

### 3. Initialize a local git repository
```bash
git init
```
Turns the current directory into a git repo (creates a hidden `.git/` folder).

### 4. Stage files for commit
```bash
git add <file>        # stage a specific file
git add -A            # stage ALL changed/new/deleted files
```
Staging means "mark this for the next commit."

### 5. Commit staged files
```bash
git commit -m "your message here"
```
Saves a snapshot of staged files with a message describing what changed.

### 6. Create a GitHub repo and push in one command
```bash
gh repo create cs336-assignment5-alignment --public --source=. --remote=origin --push
```
- Creates the repo on GitHub
- Sets it as the `origin` remote
- Pushes your local commits

### 7. Push subsequent changes
```bash
git push
```
After the first push, subsequent pushes just need `git push`.

---

## Lesson learned: always push a complete, functional project

**What went wrong:** The first commit only included `math_baseline.py`. 
That's useless on RunPod — no data, no config, no tests.

**Rule:** Before committing, ask: "Can someone clone this repo and actually run the project?"

**Correct workflow for a new project:**
```bash
# 1. Check what .gitignore is excluding
cat .gitignore

# 2. Decide if excluded files are needed on the target machine
#    (e.g. data files needed on RunPod → remove from .gitignore)

# 3. Stage everything at once
git add -A

# 4. Verify what will be committed
git status

# 5. Commit with a meaningful message
git commit -m "Initial commit: full project with data, configs, and code"

# 6. Push
git push
```

---

## Everyday workflow (after initial setup)

```bash
# Check what has changed
git status

# Stage your changes
git add cs336_alignment/my_new_file.py

# Commit
git commit -m "Add SFT training script"

# Push to GitHub
git push
```

---

## On RunPod: clone and set up

```bash
git clone https://github.com/XiaoruiWang/cs336-assignment5-alignment.git
cd cs336-assignment5-alignment
uv sync
```

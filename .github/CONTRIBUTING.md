# Contributing

## Branch Structure

```
vpy-toolkit          ← shared infrastructure (models/, utils/, patches/)
├── feature/*        ← feature development branches
└── main             ← stable releases
```

## Rules

1. **Shared code** (`models/`, `utils/`, `patches/`):
   - Edit only on `vpy-toolkit`
   - Changes require maintainer review (CODEOWNERS)
   - Feature branches inherit via merge

2. **Feature development**:
   - Branch from `vpy-toolkit`: `git checkout -b feature/my-feature vpy-toolkit`
   - Do NOT edit shared paths directly on feature branches
   - Merge `vpy-toolkit` into your feature branch to get updates

3. **Submitting changes**:
   - Open PR to appropriate branch
   - Shared code → PR to `vpy-toolkit`
   - Feature work → PR to that feature branch

## Local Hook

A pre-commit hook warns if you edit protected paths on feature branches:
```bash
# Reinstall after clone:
cp .github/hooks/pre-commit .git/hooks/
chmod +x .git/hooks/pre-commit
```

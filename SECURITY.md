# Security Policy

## Reporting Security Vulnerabilities

If you discover a security vulnerability in this project, please report it privately to help us fix it before public disclosure.

**Please do NOT:**
- Open a public GitHub issue for security vulnerabilities
- Disclose the vulnerability publicly before it has been addressed

**Please DO:**
- Email the repository owner with details of the vulnerability
- Provide sufficient information to reproduce the issue
- Allow reasonable time for the issue to be addressed before public disclosure

## Security Best Practices for Deployment

### 1. Secrets Management

**NEVER commit secrets to the repository.** This includes:
- API keys (ZYTE_API_KEY)
- Authentication passwords (API_PASSWORD)
- Access tokens (BLOB_READ_WRITE_TOKEN)
- Proxy credentials (DATACENTER_PROXY)

**Always use environment variables or secret management systems:**
- For local development: Use a `.env` file (which is gitignored)
- For Kubernetes: Use Kubernetes Secrets (as shown in `deploy/prod.yaml`)
- For other platforms: Use the platform's secret management service

### 2. Environment Variable Setup

Copy `.env.example` to `.env` and fill in your actual credentials:

```bash
cp .env.example .env
# Edit .env with your actual credentials
# NEVER commit the .env file
```

Verify `.env` is in your `.gitignore`:
```bash
grep -q "^\.env$" .gitignore && echo "✓ .env is gitignored" || echo "✗ WARNING: .env is NOT gitignored!"
```

### 3. API Password Security

- Use a strong, randomly generated password for `API_PASSWORD`
- Generate a secure password with: `openssl rand -base64 32`
- Rotate credentials regularly
- Use different credentials for different environments (dev, staging, prod)

### 4. Proxy Credentials

- Keep proxy credentials (ZYTE_API_KEY, DATACENTER_PROXY) confidential
- These credentials provide access to paid services and should be protected
- Monitor usage to detect unauthorized access

### 5. Vercel Blob Token

- Treat BLOB_READ_WRITE_TOKEN as highly sensitive
- This token has write access to your Vercel Blob storage
- Rotate tokens if you suspect they may have been compromised

### 6. Kubernetes Deployment Security

When deploying to Kubernetes:

1. **Create secrets before deployment:**
```bash
kubectl create secret generic slides-extractor-secrets \
  --from-literal=ZYTE_API_KEY=your_actual_key \
  --from-literal=BLOB_READ_WRITE_TOKEN=your_actual_token \
  --from-literal=API_PASSWORD=your_actual_password \
  --from-literal=DATACENTER_PROXY=your_actual_proxy
```

2. **Never commit production secrets to deploy/prod.yaml**
   - The deployment manifest references secrets by name
   - Actual secret values should only exist in Kubernetes secrets

3. **Use RBAC to restrict secret access:**
   - Limit which users/service accounts can read secrets
   - Use namespace isolation for different environments

### 7. Network Security

- The ingress configuration in `deploy/prod.yaml` exposes the service publicly
- Consider adding additional authentication/authorization layers
- Use TLS certificates (cert-manager is configured in the example)
- Consider IP whitelisting or VPN access for sensitive deployments

### 8. Logging Security

- Do not log sensitive information (API keys, passwords, tokens)
- Review logs before sharing them publicly
- The codebase uses proper logging practices, but be cautious when debugging

### 9. Dependency Security

Regularly update dependencies to patch security vulnerabilities:

```bash
# Update dependencies
uv sync --upgrade

# Check for known vulnerabilities (requires safety)
pip install safety
safety check
```

### 10. Pre-commit Hooks (Recommended)

Install pre-commit hooks to catch potential security issues:

```bash
pip install pre-commit detect-secrets
# Create .pre-commit-config.yaml with security checks
pre-commit install
```

Example `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']
```

## Security Checklist for Public Repositories

Before making a repository public:

- [ ] Remove all hardcoded credentials from source code
- [ ] Audit git history for accidentally committed secrets (use tools like `git-secrets` or `truffleHog`)
- [ ] Add `.env.example` with placeholder values
- [ ] Verify `.env` is in `.gitignore`
- [ ] Remove or redact production-specific configurations (domains, IPs)
- [ ] Add security documentation (this file)
- [ ] Enable GitHub secret scanning
- [ ] Set up dependabot for security updates
- [ ] Review CI/CD workflows for exposed secrets

## Known Security Considerations

### Test Files
- Test files contain hardcoded test credentials (e.g., "testpassword")
- These are marked with `# noqa: S105` and `# noqa: S106` comments
- These are ONLY for testing and should NEVER be used in production

### Production Domain
- The example deployment configuration references a specific domain
- Update `deploy/prod.yaml` with your own domain before deploying
- Do not use the example domain in production

## Security Updates

This project follows these security practices:
- Regular dependency updates
- Security-focused code linting with ruff
- Type checking with mypy to prevent type-related vulnerabilities
- Non-root Docker user for container security

## Compliance

Depending on your use case, you may need to consider:
- GDPR compliance if processing personal data
- Data retention policies for downloaded content
- Terms of service compliance for YouTube and third-party services
- Copyright and licensing considerations for downloaded content

## Questions?

If you have questions about security practices for this project, please open a discussion (not an issue) on GitHub.

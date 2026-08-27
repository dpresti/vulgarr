# Security Policy

Vulgarr is a self-hosted app that touches your media library and can be reached
over the network (webhooks from Sonarr/Radarr/Bazarr, and the web UI itself if
you expose it beyond your LAN). Reports of real vulnerabilities are taken
seriously.

## Supported versions

There are no maintained release branches -- only the latest commit on `main`
is supported. If you're running an older commit, update before reporting an
issue; it may already be fixed.

## Reporting a vulnerability

**Please don't open a public GitHub issue for a security vulnerability.**
Doing so discloses it to everyone self-hosting this app before a fix exists.

Instead, use GitHub's [private vulnerability
reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing/privately-reporting-a-security-vulnerability):
open the **Security** tab on this repo and click **Report a vulnerability**.
That opens a private conversation with the maintainer that nobody else can
see, and can turn into a GitHub Security Advisory (with CVE, if warranted)
once it's resolved.

Include what you'd include in any good bug report: the affected component
(webhook handling, auth, file/path handling, etc.), steps to reproduce, and
what an attacker could actually do with it.

## Scope

Realistic areas of concern for this project:

- The webhook endpoints (`/webhooks/...`) and their token check
- Basic Auth (`auth_enabled` in Settings) and session/cookie handling
- Path handling around media files (traversal, symlink issues)
- Anything that could let one self-hosted instance be tricked into running
  arbitrary commands or reading/writing files outside its media root

Vulnerabilities in third-party dependencies (FastAPI, ffmpeg, etc.) should
generally be reported upstream instead, unless this project is using them in
a way that specifically creates the vulnerability.

# Security

## Supported versions

Only the latest release on [PyPI](https://pypi.org/project/ovat/) gets
security fixes. If you are on an older version, upgrade first
(`pip install -U ovat`) and check whether the problem is still there.

## Reporting a problem

Please do not open a public issue for a security problem.

Report it privately through GitHub instead: open the
[Security tab](https://github.com/Lagmator22/ovat/security) of this
repository and choose **Report a vulnerability**. Only the maintainer can
see the report.

Helpful things to include:

- the OVAT version (`ovat --version`) and your OS
- the workflow file or command that shows the problem, cut down as far as
  you can
- what happens, and what you expected

## What to expect

- You should get a first reply within a week.
- If the report is confirmed, the fix ships in a new release. The advisory
  is published once that release is on PyPI, and you are credited unless
  you ask not to be.
- If it is not a security problem, you will be told why, and it can move to
  a normal issue.

## What is not a vulnerability

A workflow file is code, not data. A `tools:` entry with `type: mcp_stdio`
starts the command it names, with your user's permissions. Running a
workflow from someone you do not trust is like running their script. That
is how it is designed to work, not a bug. Read a workflow before you run
it.

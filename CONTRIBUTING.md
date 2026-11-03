# Contributing

## Introduction

We welcome contributions via pull requests, including bug fixes, new features,
and documentation improvements. Keep pull requests focused, describe your changes,
and include relevant tests or explain how you verified the changes.

This project is licensed under the [Apache License, Version 2.0](LICENSE.md).
Contributions must be submitted under this license. We use the Developer
Certificate of Origin (DCO), explained below, to document contributors' right to
submit their work.

## Developer Certificate of Origin

The [Developer Certificate of Origin, Version 1.1](https://developercertificate.org/)
is a certification that you created a contribution or otherwise have the right to
submit it under the applicable open source license. Read the original document
before signing off your contributions:


> Developer Certificate of Origin
> Version 1.1
> 
> Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
> 
> Everyone is permitted to copy and distribute verbatim copies of this
> license document, but changing it is not allowed.
> 
> 
> Developer's Certificate of Origin 1.1
> 
> By making a contribution to this project, I certify that:
> 
> (a) The contribution was created in whole or in part by me and I
>     have the right to submit it under the open source license
>     indicated in the file; or
> 
> (b) The contribution is based upon previous work that, to the best
>     of my knowledge, is covered under an appropriate open source
>     license and I have the right under that license to submit that
>     work with modifications, whether created in whole or in part
>     by me, under the same open source license (unless I am
>     permitted to submit under a different license), as indicated
>     in the file; or
> 
> (c) The contribution was provided directly to me by some other
>     person who certified (a), (b) or (c) and I have not modified
>     it.
> 
> (d) I understand and agree that this project and the contribution
>     are public and that a record of the contribution (including all
>     personal information I submit with it, including my sign-off) is
>     maintained indefinitely and may be redistributed consistent with
>     this project or the open source license(s) involved.


Only sign off work you are authorized to submit. If you contribute work created
for an employer or another party, ensure you have any necessary permission.
Signing off does not transfer copyright ownership; it records your DCO
certification and your right to contribute under the applicable license.

## Signing Off

Every commit in a pull request must include a `Signed-off-by` line at the end of
its commit message. By adding this line, you certify compliance with DCO 1.1.
Use your real name and an email address you control. Nicknames, pseudonyms, and
anonymous contributions are not accepted.

For example, using a fictional contributor's name (`John Doe`) and email 
address (`john.doe@example.com`):

```text
Fix evaluation error handling

Signed-off-by: John Doe <john.doe@example.com>
```

Before committing, configure Git with your own real name and email address.
These commands set your identity for this repository; add `--global` to apply
them to all your repositories instead:

```sh
git config user.name "John Doe"
git config user.email "john.doe@example.com"
```

Git can add the sign-off automatically using `-s` (or `--signoff`):

```sh
git commit -s -m "Fix evaluation error handling"
```

If you forgot to sign off your most recent commit, and it is your own work,
amend it without changing its message:

```sh
git commit --amend --no-edit --signoff
```

Amending changes the commit ID. If the commit has already been pushed to your
pull request branch, coordinate with anyone sharing that branch before updating
it with `git push --force-with-lease`. If several commits lack sign-offs, amend
each affected commit, for example using an interactive rebase; signing off only
the latest commit does not cover earlier commits. Do not add someone else's
sign-off on their behalf.

A DCO sign-off is a commit-message declaration, not a cryptographic signature.
Git's `-s` option adds the required declaration; `-S` signs a commit
cryptographically and does not replace the DCO sign-off.

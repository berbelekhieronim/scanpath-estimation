# tools/

## Checking what your hosting supports

We need to know one thing: **does your host have `sqlite` or only `mysql`?**
That picks which storage implementation the app uses. Either works — SQLite is
simpler (no database setup), MySQL is guaranteed present on any PHP host.

**This is not urgent.** If it's a hassle, skip it: we default to MySQL, which
every PHP shared host provides, and switch to SQLite later if it turns out to be
available. Nothing in the build waits on this.

### Easiest way — your host's file manager (no software to install)

1. Log in to your Zenbox control panel in a browser.
2. Find the **File Manager** (most panels have one; it may be called "Menedżer plików").
3. Open the public web folder. It is usually named `public_html`, `www`, or
   `domains/yourdomain.pl/public_html` — the one whose contents appear when you
   visit your domain.
4. Create a new file called `check.php`.
5. Paste in the two lines from `check_host_mini.php` in this folder.
6. Save, then visit `https://yourdomain.pl/check.php` in a browser.
7. Note what it says, then **delete the file**.

### Alternative — FTP client

Same thing, using the FTP details you already have:

1. Install [FileZilla](https://filezilla-project.org/) (free, works on macOS).
2. Connect with your Zenbox FTP host, username and password.
3. On the right-hand side, open the public web folder (see step 3 above).
4. Drag `check_host_mini.php` across from the left-hand side.
5. Visit `https://yourdomain.pl/check_host_mini.php`.
6. Note the result, then delete the file from the server.

### What the answer means

The second line lists the database drivers PHP can use.

| If the list contains | Then |
|---|---|
| `sqlite` | Use **SqliteStorage** — simplest, nothing to set up |
| only `mysql` (no `sqlite`) | Use **MysqlStorage** — create a database in the Zenbox panel and note the host, name, user and password |
| neither | Unlikely. Send me what it printed |

`check_host.php` in this folder is the fuller version — it also tests whether
SQLite can genuinely open a file database, checks write permissions and reports
upload limits. Use it if the short one is ambiguous. Same upload steps.

### A third option

Your control panel may list PHP extensions directly, often under something like
"PHP configuration" or "Select PHP version". If you find a list of extensions
there, look for `pdo_sqlite` — that answers the question with no file upload at
all. I don't know Zenbox's panel layout, so I can't point you at the exact menu.
